import os 

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import jax.numpy as jnp
from conformer_diffs.pytorch.diff_utils.torch2jax_utils import Torch2Jax, flatten
import functools
from conformer_diffs.pytorch.model import ConformerEncoderDecoder, ConformerConfig
from flax.training import checkpoints as flax_checkpoints

from conformer_diffs.jax_impl.model import ConformerConfig as JaxConformerConfig
from conformer_diffs.jax_impl.model import Conformer as JaxConformer

MAX_INPUT_LENGTH = 320000

def value_transform(k, value, jax_value):
  k_str = ''.join(k).lower()
  if ('conv' in k_str and 'kernel' in k_str) or \
    ('embedding' in k_str and 'kernel' in k_str):
    if 'transpose' in k_str:
      # Assumes 2D ConvTranspose with stride equal to kernel_size.
      return value.reshape(value.shape[0], value.shape[1],
                           -1).flip(-1).permute(2, 0,
                                                1).reshape(*jax_value.shape)
    else:
      rank = len(value.shape)
      if rank == 3:
        value = value.permute(2, 1, 0)
      elif rank == 4:
        value = value.permute(2, 3, 1, 0)
      elif rank == 2:
        value = value.t()
  elif 'attention' in k_str and 'kernel' in k_str:
    value = value.t().reshape(*list(jax_value.shape))
  elif 'attention' in k_str and 'bias' in k_str:
    value = value.reshape(*list(jax_value.shape))
  elif ('dense' in k_str and 'kernel' in k_str) or \
    ('lstm' in k_str and 'kernel' in k_str) or \
    ('head' in k_str and 'kernel' in k_str) or \
    ('pre_logits' in k_str and 'kernel' in k_str):
    value = value.t()
  return value

def key_transform(k):
  new_key = []
  for i in k:
    if 'ModuleList' in i:
      continue
    if 'Linear' in i:
      if 'NonDynamicallyQuantizableLinear' in i:
        i = 'out'
      else:
        i = i.replace('Linear', 'Dense')
    elif 'Conv1d' in i:
      i = i.replace('Conv1d', 'Conv')
    elif 'MHSAwithQS' in i:
      i = i.replace('MHSAwithQS', 'SelfAttention')
    elif 'weight' in i:
      i = i.replace('weight', 'kernel')
    new_key.append(i)
  return tuple(new_key)


def sd_transform(sd):
  out = {}
  for k in sd:
    if 'Attention' in ''.join(k):
      if 'in_proj' in k[-1]:
        new_key = k[:-1]
        chunks = sd[k].chunk(3)
        for t, c in zip(['query', 'key', 'value'], chunks):
          out[new_key + (t, k[-1].split('_')[-1])] = c
      else:
        out[k] = sd[k]
    else:
      out[k] = sd[k]
  return out


def initialize(m):
  if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
    init.xavier_uniform_(m.weight)
    if m.bias is not None:
      init.constant_(m.bias, 0)
  elif isinstance(m, nn.MultiheadAttention):
    init.xavier_uniform_(m.in_proj_weight)
  for i in m.children():
    initialize(i)

def init_model_fn_jax(rng):
    return

def init_model_fn_torch(rng):
    """Conformer model init function.

    Here we use dropout_rate as feed_forward_dropout_rate, and aux_dropout_rate
    as input_dropout_rate.
    """
    torch.random.manual_seed(rng[0])
    model = ConformerEncoderDecoder(
        ConformerConfig(
            feed_forward_dropout_rate=0.0,
            input_dropout_rate=0.0)).eval()
    # Run model once to initialize lazy layers.
    t = MAX_INPUT_LENGTH
    wave = torch.randn((2, t))
    pad = torch.zeros_like(wave)
    _ = model(wave, pad)
    initialize(model)
    return model

def load_pyt_batch():
  # Loading 1 real batch 
  RANK = 0
  sharded_padded_batch = np.load('sharded_padded_batch.npz')

  inputs, input_paddings = sharded_padded_batch['inputs']
  targets, target_paddings = sharded_padded_batch['targets']

  inputs = inputs.reshape(256, -1)[RANK*32: (RANK + 1)*32]
  input_paddings = input_paddings.reshape(256, -1)[RANK*32: (RANK + 1)*32]
  targets = targets.reshape(256, -1)[RANK*32: (RANK + 1)*32]
  target_paddings = target_paddings.reshape(256, -1)[RANK*32: (RANK + 1)*32]

  sharded_padded_batch = {
      'inputs': (torch.from_numpy(inputs), torch.from_numpy(input_paddings)),
      'targets': (torch.from_numpy(targets), torch.from_numpy(target_paddings))
  }
  return sharded_padded_batch


def load_jax_batch():
  # Loading 1 real batch 
  RANK = 0
  sharded_padded_batch = np.load('sharded_padded_batch.npz')

  inputs, input_paddings = sharded_padded_batch['inputs']
  targets, target_paddings = sharded_padded_batch['targets']

  print('loaded librispeech sharded padded batch')
  print('inputs shape = ', inputs.shape)
  print('input paddings shape = ', input_paddings.shape)
  print('targets shape = ', targets.shape)
  print('target_paddings shape = ', target_paddings.shape)

  sharded_padded_batch = {
      'inputs': (inputs[RANK], input_paddings[RANK]),
      'targets': (targets[RANK], target_paddings[RANK])
  }

  return sharded_padded_batch

def pyt_ctcloss(logits, logits_paddings, targets, target_paddings):
  logprobs = torch.log_softmax(logits, dim=-1)
  input_lengths = torch.einsum('bh->b', 1 - logits_paddings).long()
  target_lengths = torch.einsum('bh->b', 1 - target_paddings).long()
  ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')

  per_seq_loss = ctc_loss(
      logprobs.permute(1, 0, 2),
      targets.long(),
      input_lengths,
      target_lengths).sum()
  l = target_lengths.sum().to(per_seq_loss)
  return per_seq_loss/l

def jax_ctcloss(logits, logit_paddings, targets, target_paddings):
  import flax.linen as nn
  import optax
  logprobs = nn.log_softmax(logits)
  per_seq_loss = optax.ctc_loss(logprobs,
                                logit_paddings,
                                targets,
                                target_paddings)
  normalizer = jnp.sum(1 - target_paddings)
  normalized_loss = jnp.sum(per_seq_loss) / jnp.maximum(normalizer, 1)
  return normalized_loss

if __name__ == '__main__':
    # pylint: disable=locally-disabled, not-callable
    rng = jax.random.PRNGKey(0)
    print('Initializing PyTorch model')
    torch_model = init_model_fn_torch(rng)

    config = JaxConformerConfig(feed_forward_dropout_rate=0.0, input_dropout_rate=0.0)
    model_class = JaxConformer(config)
    params_rng, dropout_rng = jax.random.split(rng, 2)

    model_init_fn = jax.jit(functools.partial(model_class.init, train=False))
    input_shape = [(320000,), (320000,)]
    fake_input_batch = [np.zeros((2, *x), jnp.float32) for x in input_shape]

    print('Initializing JAX model.')
    vars = model_init_fn({'params': params_rng, 'dropout': dropout_rng}, *fake_input_batch)
    jax_batch_stats, jax_params = vars.pop('params')
    jax_params = jax_params.unfreeze()
    t2j = Torch2Jax(torch_model=torch_model, jax_model=jax_params)
    t2j.key_transform(key_transform)
    t2j.sd_transform(sd_transform)
    t2j.value_transform(value_transform)
    t2j.diff()
    t2j.update_jax_model()

    jax_batch  = load_jax_batch()
    pyt_batch  = load_pyt_batch()
    # wave = torch.randn(2, 320000)
    # pad = torch.zeros_like(wave)
    # pad[0, 200000:] = 1

    # jax_batch = {'inputs': (wave.detach().numpy(), pad.detach().numpy())}
    # pyt_batch = {'inputs': (wave, pad)}

    (out_j, outp_j), _ = model_class.apply({'params':jax_params,**jax_batch_stats},jax_batch['inputs'][0],jax_batch['inputs'][1],train=True,mutable=[
      'batch_stats'
    ])
    torch_model.train()
    out_p, outp_p = torch_model(pyt_batch['inputs'][0], pyt_batch['inputs'][1])
    
    print(pyt_ctcloss(out_p, outp_p, pyt_batch['targets'][0], pyt_batch['targets'][1]))
    print(jax_ctcloss(out_j, outp_j, jax_batch['targets'][0], jax_batch['targets'][1]))

    out_j = out_j*(1-outp_j[:,:,None])
    out_p = out_p*(1-outp_p[:,:,None])
    # out_j = outp_j 
    # out_p = outp_p 

    print(np.abs(out_p.detach().numpy() - np.array(out_j)).reshape(2,-1).max(axis=1))
    print(np.abs(out_p.detach().numpy() - np.array(out_j)).reshape(2,-1).sum(axis=1))
    print(np.abs(out_p.detach().numpy() - np.array(out_j)).reshape(2,-1).mean(axis=1))

    flax_checkpoints.save_checkpoint('ckpts', target={'params':jax_params,'batch_stats':jax_batch_stats}, step=0, overwrite=True)
    torch.save(torch_model.state_dict(), 'ckpts/torch_model_weights.pt')


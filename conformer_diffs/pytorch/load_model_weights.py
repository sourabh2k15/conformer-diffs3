import torch
import jax

import flax
from flax.training import checkpoints as flax_checkpoints

if __name__ == '__main__':
    restored = flax_checkpoints.restore_checkpoint(
      'ckpts', target=None, prefix='checkpoint')

    torch_model = torch.load('ckpts/torch_model_weights.pt')

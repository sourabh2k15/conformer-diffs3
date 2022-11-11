import numpy as np

sharded_padded_batch = np.load('sharded_padded_batch.npz')

inputs, input_paddings = sharded_padded_batch['inputs']
targets, target_paddings = sharded_padded_batch['targets']

print(inputs.shape)
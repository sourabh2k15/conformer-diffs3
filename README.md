# Setup

## VM Setup:
We've run all the code on a GCP Ubuntu 20.04 VM with 8 V100 GPU, Cuda 11.7, CuDNN 8.5
setup script can be seen here : 
https://gist.githubusercontent.com/sourabh2k15/9dbadd0f5ca35568ca210ee4cb3b19c1/raw/be988210438f91eda85fc608a9c71744a4c8af89/vm_setup.sh

we simply wget above script and run to install required elements. 

## Code Setup 
git clone https://github.com/sourabh2k15/conformer-diffs.git

cd conformer-diffs
pip3 install -e .

cd conformer_diffs


## Data Setup 

1) download real batch using :
```
wget https://transfer.sh/oanN8q/sharded_padded_batch.npz
```
this will give you one sharded batch of real librispeech data with size 256, this batch has input_paddings and target paddings includes.

dimensions:

inputs - (8, 32, 320000)

input_paddings - (8, 32, 320000)

targets - (8, 32, 256)

target_paddings - (8, 32, 256)

2) create jax and pytorch checkpoints by running the following script 

```
cd pytorch
python3 dump_model_weights.pygit rm --cached

```

this should dump same model weights for JAX and PyTorch in ckpts directory inside torch/

# Reproduction

## JAX
Command to run the jax example : 
```
python3 jax/jax_e2e.py
```

## PyTorch
Command to run the torch example : 
```
torchrun --standalone --nnodes 1 --nproc_per_node 8 pytorch/torch_e2e.py
```

# Results: 


1) JAX logs 

```

```

2) PyTorch logs

```


```

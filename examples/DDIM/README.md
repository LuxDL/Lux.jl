# Denoising Diffusion Implicit Model (DDIM)

[Lux.jl](https://github.com/LuxDL/Lux.jl) implementation of Denoising Diffusion Implicit Models ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502)).

The implementation follows [the Keras example](https://keras.io/examples/generative/ddim/).

The model generates images from Gaussian noises by denoising iteratively.

TODO: add image

# Usage

Install Julia and instantiate `Project.toml`.

Following scripts are tested on a single NVIDIA V100 instance with 32GB of GPU memory. You
may need to adjust the batch size and learning rate for your environment.

## Dataset

We use the dataset from [102 Category Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
The user is prompted to download the dataset when running the code for the first time.
The dataset is cached for subsequent runs.

## Training

```bash
julia --project main.jl \
    --dataset-dir oxford_flower_102 \ # path to dataset directory containing image files
    --epochs 25 \
    --image-size 96 \
    --batchsize 64 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
    --val-diffusion-steps 80 \
    --output-dir output/train # path to save checkpoint and images
```

This code runs in about XXXX minutes in a single NVIDIA V100 instance with 32GB of GPU
memory. We recommend running the code for atleast 80 epochs to get good results.

## Image generation

```bash
julia --project main.jl \
    --checkpoint output/ckpt/checkpoint_25.bson \ # path to checkpoint
    --image-size 96 \
    --num-images 10 \
    --diffusion-steps 80 \
    --output-dir output/generate # path to save images
```

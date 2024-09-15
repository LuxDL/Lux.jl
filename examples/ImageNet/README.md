# Imagenet Training using Lux

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on
the ImageNet dataset.

## Requirements

* Install [julia](https://julialang.org/)
* In the Julia REPL instantiate the `Project.toml` in the parent directory
* Download the ImageNet dataset from http://www.image-net.org/
  - Then, move and extract the training and validation images to labeled subfolders, using
    [this shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)

## Training

To train a model, run `main.jl` with the necessary parameters. See
[Boltz documentation](https://luxdl.github.io/Boltz.jl/stable/) for the model configuration.

```bash
julia --startup=no --project=examples/ImageNet -t auto examples/ImageNet/main.jl \
  --model-name="VGG" \
  --depth=19 \
  --train-batchsize=256 \
  --val-batchsize=256 \
  --optimizer-kind="sgd" \
  --learning-rate=0.01 \
  --base-path="/home/avik-pal/data/ImageNet/"


julia --startup=no --project=examples/ImageNet -t auto examples/ImageNet/main.jl \
  --model-name="ViT" \
  --model-kind="tiny" \
  --train-batchsize=256 \
  --val-batchsize=256 \
  --optimizer-kind="sgd" \
  --learning-rate=0.01 \
  --base-path="/home/avik-pal/data/ImageNet/"
```

## Distributed Data Parallel Training

Setup [MPI.jl](https://juliaparallel.org/MPI.jl/).
If your system has functional NCCL we will use it for all CUDA communications. Otherwise, we
will use MPI for all communications.

```bash
mpiexecjl -np 4 julia --startup=no --project=examples/ImageNet -t auto\
  examples/ImageNet/main.jl \
  --model-name="ViT" \
  --model-kind="tiny" \
  --train-batchsize=256 \
  --val-batchsize=256 \
  --optimizer-kind="sgd" \
  --learning-rate=0.01 \
  --base-path="/home/avik-pal/data/ImageNet/"
```

## Usage

```bash
  main

Usage

  main [options] [flags]

Options

  --seed <0::Integer>
  --model-name <String>
  --model-kind <nokind::String>
  --depth <-1::Int>
  --base-path <::String>
  --train-batchsize <64::Int>
  --val-batchsize <64::Int>
  --image-size <-1::Int>
  --optimizer-kind <sgd::String>
  --learning-rate <0.01::Float32>
  --momentum <0.0::Float32>
  --weight-decay <0.0::Float32>
  --scheduler-kind <step::String>
  --cycle-length <50000::Int>
  --damp-factor <1.2::Float32>
  --lr-step-decay <0.1::Float32>
  --lr-step <[100000...::Vector{Int64}>
  --expt-id <::String>
  --expt-subdir <#= /home...::String>
  --resume <::String>
  --total-steps <800000::Int>
  --evaluate-every <10000::Integer>
  --print-frequency <100::Integer>

Flags

  --pretrained
  --nesterov
  --evaluate
  -h, --help                                                Print this help message.
  --version                                                 Print version.
```

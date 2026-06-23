# NanoGPT using Lux & Reactant

## Requirements

* Install [julia](https://julialang.org/)
* In the Julia REPL instantiate the `Project.toml` in the parent directory

## Training

To train a model, run `main.jl` with the necessary parameters.

```bash
julia --startup=no --project=examples/NanoGPT --threads=auto examples/NanoGPT/main.jl
```

## Inference

To run inference on a trained model, run `main.jl` with the necessary parameters.

```bash
julia --startup=no --project=examples/NanoGPT --threads=auto examples/NanoGPT/main.jl \
  --inference \
  --model-path=<path to model checkpoint>
```

## Usage

```bash
  main

Usage

  main [options] [flags]

Options

  --n-embed <64::Int>
  --n-hidden <256::Int>
  --n-heads <4::Int>
  --qk-dim <16::Int>
  --v-dim <16::Int>
  --n-layers <6::Int>
  --sequence-length <64::Int>
  --batchsize <128::Int>
  --dropout-rate <0.0::Float32>
  --test-split <0.1::Float64>
  --lr <0.01::Float64>
  --epochs <100::Int>
  --model-path <::String>
  --seed <::Union{String, Vector{String}}>
  --output-length <1024::Int>

Flags

  --inference
  -h, --help                                                Print this help message.
  --version                                                 Print version.
```

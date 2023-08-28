# Getting Started

## Installation

Install [Julia v1.6 or above](https://julialang.org/downloads/). Lux.jl is available through
the Julia package manager. You can enter it by pressing `]` in the REPL and then typing

```julia
pkg> add Lux
```

Alternatively, you can also do

```julia
import Pkg; Pkg.add("Lux")
```

:::tip

The Julia Compiler is always improving. As such, we recommend using the latest stable
version of Julia instead of the LTS.

:::

## Quickstart

:::tip PRE-REQUISITES

You need to install `Optimisers` and `Zygote` if not done already.
`Pkg.add(["Optimisers", "Zygote"])`

:::

```@example quickstart
using Lux, Random, Optimisers, Zygote
# using LuxCUDA, LuxAMDGPU, Metal # Optional packages for GPU support
```

We take randomness very seriously

```@example quickstart
# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)
```

Build the model

```@example quickstart
# Construct the layer
model = Chain(Dense(128, 256, tanh), Chain(Dense(256, 1, tanh), Dense(1, 10)))
```

Models don't hold parameters and states so initialize them. From there on, we just use our
standard AD and Optimisers API.

```@example quickstart
# Get the device determined by Lux
device = gpu_device()

# Parameter and State Variables
ps, st = Lux.setup(rng, model) .|> device

# Dummy Input
x = rand(rng, Float32, 128, 2) |> device

# Run the model
y, st = Lux.apply(model, x, ps, st)

# Gradients
## Pullback API to capture change in state
(l, st_), pb = pullback(p -> Lux.apply(model, x, p, st), ps)
gs = pb((one.(l), nothing))[1]

# Optimization
st_opt = Optimisers.setup(Adam(0.0001f0), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)
```

## Additional Packages

`LuxDL` hosts various packages that provide additional functionality for Lux.jl. All
packages mentioned in this documentation are available via the Julia General Registry.

You can install all those packages via `import Pkg; Pkg.add(<package name>)`.

## GPU Support

GPU Support for Lux.jl requires loading additional packages:

* [`LuxCUDA.jl`](https://github.com/LuxDL/LuxCUDA.jl) for CUDA support.
* [`LuxAMDGPU.jl`](https://github.com/LuxDL/LuxAMDGPU.jl) for AMDGPU support.
* [`Metal.jl`](https://github.com/JuliaGPU/Metal.jl) for Apple Metal support.

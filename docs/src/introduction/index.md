# [Getting Started](@id getting-started)

## Installation

Install [Julia v1.9 or above](https://julialang.org/downloads/). Lux.jl is available through
the Julia package manager. You can enter it by pressing `]` in the REPL and then typing

```julia
pkg> add Lux
```

Alternatively, you can also do

```julia
import Pkg; Pkg.add("Lux")
```

!!! tip

    The Julia Compiler is always improving. As such, we recommend using the latest stable
    version of Julia instead of the LTS.

## Quickstart

!!! tip "Pre-Requisites"

    You need to install `Optimisers` and `Zygote` if not done already.
    `Pkg.add(["Optimisers", "Zygote"])`

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

## Defining Custom Layers

```@example custom_compact
using Lux, Random, Optimisers, Zygote
# using LuxCUDA, LuxAMDGPU, Metal # Optional packages for GPU support
import Lux.Experimental: @compact
```

We will define a custom MLP using the `@compact` macro. The macro takes in a list of
parameters, layers and states, and a function defining the forward pass of the neural
network.

```@example custom_compact
n_in = 1
n_out = 1
nlayers = 3

model = @compact(w1=Dense(n_in, 128),
    w2=[Dense(128, 128) for i in 1:nlayers],
    w3=Dense(128, n_out),
    act=relu) do x
    embed = act(w1(x))
    for w in w2
        embed = act(w(embed))
    end
    out = w3(embed)
    return out
end
```

We can initialize the model and train it with the same code as before!

```@example custom_compact
ps, st = Lux.setup(Xoshiro(0), model)

model(randn(n_in, 32), ps, st)  # 1×32 Matrix as output.

x_data = collect(-2.0f0:0.1f0:2.0f0)'
y_data = 2 .* x_data .- x_data .^ 3
st_opt = Optimisers.setup(Adam(), ps)

for epoch in 1:1000
    global st  # Put this in a function in real use-cases
    (loss, st), pb = Zygote.pullback(ps) do p
        y, st_ = model(x_data, p, st)
        return sum(abs2, y .- y_data), st_
    end
    gs = only(pb((one(loss), nothing)))
    epoch % 100 == 1 && println("Epoch: $(epoch) | Loss: $(loss)")
    Optimisers.update!(st_opt, ps, gs)
end
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

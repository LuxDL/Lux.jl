# Lux

`Lux` is a julia deep learning framework which decouples models and parameterization using deeply nested named tuples.

- Functional Layer API -- Pure Functions and Deterministic Function Calls.
- No more implicit parameterization -- `Zygote.Params`. Everything is a `NamedTuple`.
- Compiler and AD-friendly Neural Networks

# Installation

Install [julia v1.6 or above](https://julialang.org/downloads/).

```julia
using Pkg
Pkg.add("Lux")
```

# Quick Example

```julia
using Lux, Random, Optimisers, Zygote
```

We take randomness very seriously

```julia
# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)
```

Build the model

```julia
# Construct the layer
model = Chain(
    BatchNorm(128),
    Dense(128, 256, tanh),
    BatchNorm(256),
    Chain(
        Dense(256, 1, tanh),
        Dense(1, 10)
    )
)
```

Models don't hold parameters and states so initialize them. From there on, we just use our standard AD and Optimisers API.

```julia
# Parameter and State Variables
ps, st = Lux.setup(rng, model) .|> gpu

# Dummy Input
x = rand(rng, Float32, 128, 2) |> gpu

# Run the model
y, st = Lux.apply(model, x, ps, st)

# Gradients
gs = gradient(p -> sum(Lux.apply(model, x, p, st)[1]), ps)[1]

# Optimization
st_opt = Optimisers.setup(Optimisers.ADAM(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)
```

# Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@misc{pal2022lux,
    author = {Pal, Avik},
    title = {Lux: Explicit Parameterization of Deep Neural Networks in Julia},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/avik-pal/Lux.jl/}}
}
```

Also consider starring [our github repo](https://github.com/avik-pal/Lux.jl/)
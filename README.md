# Lux 🔥

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/)

[![CI](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml)
[![CI Nightly](https://github.com/avik-pal/Lux.jl/actions/workflows/CINightly.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CINightly.yml)
[![codecov](https://codecov.io/gh/avik-pal/Lux.jl/branch/main/graph/badge.svg?token=IMqBM1e3hz)](https://codecov.io/gh/avik-pal/Lux.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Lux)](https://pkgs.genieframework.com?packages=Lux)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

The 🔥 Deep Learning Framework

## Installation

```julia
] add Lux
```

## Getting Started

```julia
using Lux, Random, Optimisers, Zygote

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

# Construct the layer
model = Chain(BatchNorm(128), Dense(128, 256, tanh), BatchNorm(256),
              Chain(Dense(256, 1, tanh), Dense(1, 10)))

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

## Examples

Look in the [examples](/examples/) directory for self-contained usage examples. The [documentation](https://lux.csail.mit.edu/dev) has examples sorted into proper categories.

## Ecosystem

Checkout our [Ecosystem](http://lux.csail.mit.edu/dev/introduction/ecosystem/) page for more details. 

## Getting Help

For usage related questions, please use [Github Discussions](https://github.com/avik-pal/Lux.jl/discussions) or [JuliaLang Discourse (machine learning domain)](https://discourse.julialang.org/c/domain/ml/) which allows questions and answers to be indexed. To report bugs use [github issues](https://github.com/avik-pal/Lux.jl/issues) or even better send in a [pull request](https://github.com/avik-pal/Lux.jl/pulls).

## Related Projects

* [Flux.jl](https://github.com/FluxML/Flux.jl) -- We share most of the backend infrastructure with Flux ([Roadmap](https://github.com/FluxML/Flux.jl/issues/1829) hints towards making Flux explicit-parameter first)
* [Knet.jl](https://github.com/denizyuret/Knet.jl) -- One of the mature and OG Julia Deep Learning Frameworks
* [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) -- Extremely Efficient for Small Neural Networks on CPU
* [Avalon.jl](https://github.com/dfdx/Avalon.jl) -- Uses tracing based AD [Yota.jl](https://github.com/dfdx/Yota.jl)

## Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@software{pal2022lux,
    author = {Pal, Avik},
    title = {{L}ux: Explicit Parameterization of Deep Neural Networks in Julia},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/avik-pal/Lux.jl/}}
}
```

Also consider starring [our github repo](https://github.com/avik-pal/Lux.jl/)
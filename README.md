<p align="center">
    <img width="400px" src="docs/src/assets/lux-logo.svg#gh-light-mode-only"/>
    <img width="400px" src="docs/src/assets/lux-logo-dark.svg#gh-dark-mode-only"/>
</p>

<div align="center">

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/)

[![CI](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml)
[![Build status](https://img.shields.io/buildkite/ba1f9622add5978c2d7b194563fd9327113c9c21e5734be20e/main.svg?label=gpu)](https://buildkite.com/julialang/lux-dot-jl)
[![codecov](https://codecov.io/gh/LuxDL/Lux.jl/branch/main/graph/badge.svg?token=IMqBM1e3hz)](https://codecov.io/gh/LuxDL/Lux.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Lux)](https://pkgs.genieframework.com?packages=Lux)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

</div>

The 🔥 Deep Learning Framework

## Installation

```julia
] add Lux
```

## Getting Started

```julia
using Lux, Random, Optimisers, Zygote
# using LuxCUDA, LuxAMDGPU # Optional packages for GPU support

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

# Construct the layer
model = Chain(BatchNorm(128), Dense(128, 256, tanh), BatchNorm(256),
              Chain(Dense(256, 1, tanh), Dense(1, 10)))

# Get the device determined by Lux
device = gpu_device()

# Parameter and State Variables
ps, st = Lux.setup(rng, model) .|> device

# Dummy Input
x = rand(rng, Float32, 128, 2) |> device

# Run the model
y, st = Lux.apply(model, x, ps, st)

# Gradients
gs = gradient(p -> sum(Lux.apply(model, x, p, st)[1]), ps)[1]

# Optimization
st_opt = Optimisers.setup(Optimisers.Adam(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)
```

## Examples

Look in the [examples](/examples/) directory for self-contained usage examples. The [documentation](https://lux.csail.mit.edu/dev) has examples sorted into proper categories.

## Ecosystem

Checkout our [Ecosystem](http://lux.csail.mit.edu/dev/introduction/ecosystem/) page for more details. 

## Getting Help

For usage related questions, please use [Github Discussions](https://github.com/avik-pal/Lux.jl/discussions) or [JuliaLang Discourse (machine learning domain)](https://discourse.julialang.org/c/domain/ml/) which allows questions and answers to be indexed. To report bugs use [github issues](https://github.com/avik-pal/Lux.jl/issues) or even better send in a [pull request](https://github.com/avik-pal/Lux.jl/pulls).


## Package Ecosystem Structure

Structure of the packages part of the `Lux.jl` Universe[^1]: (Rounded Rectangles denote packages maintained by `Lux.jl` developers)

[^1]: These packages only constitute a subset of the ecosystem. Specifically these are the packages which the maintainers of Lux.jl have personally tested out. If you want a new package to be listed here, please open an issue.

```mermaid
flowchart LR
    subgraph Interface
        LuxCore(LuxCore)
    end
    subgraph Backend
        LuxLib(LuxLib)
        NNlib
        CUDA
    end
    subgraph ExternalML[External ML Packages]
        Flux
        Metalhead
    end
    subgraph CompViz[Computer Vision]
        Boltz(Boltz)
    end
    subgraph SciML[Scientific Machine Learning]
        DeepEquilibriumNetworks(DeepEquilibriumNetworks)
        DiffEqFlux(DiffEqFlux)
        NeuralPDE[Neural PDE: PINNs]
    end
    subgraph AD[Automatic Differentiation]
        Zygote
        Enzyme["Enzyme (experimental)"]
    end
    subgraph Dist[Distributed Training]
        FluxMPI(FluxMPI)
    end
    subgraph SerializeModels[Serialize Models]
        Serial[Serialization]
        JLD2
        BSON
    end
    subgraph Opt[Optimization]
        Optimisers
        Optimization
    end
    subgraph Parameters
        ComponentArrays
    end
    Lux(Lux)
    Parameters --> Lux
    LuxCore --> Lux
    Backend --> Lux
    Lux --> SciML
    AD --> Lux
    Lux --> Dist
    Lux --> SerializeModels
    Lux --> Opt
    Lux --> CompViz
    ExternalML -.-> CompViz
```

## Related Projects

* [Flux.jl](https://github.com/FluxML/Flux.jl) -- We share most of the backend infrastructure with Flux ([Roadmap](https://github.com/FluxML/Flux.jl/issues/1829) hints towards making Flux explicit-parameter first)
* [Knet.jl](https://github.com/denizyuret/Knet.jl) -- One of the mature and OG Julia Deep Learning Frameworks
* [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) -- Extremely Efficient for Small Neural Networks on CPU
* [Avalon.jl](https://github.com/dfdx/Avalon.jl) -- Uses tracing based AD [Yota.jl](https://github.com/dfdx/Yota.jl)

## Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@software{pal2023lux,
  author       = {Pal, Avik},
  title        = {{Lux: Explicit Parameterization of Deep Neural Networks in Julia}},
  month        = April,
  year         = 2023,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {v0.4.50},
  doi          = {10.5281/zenodo.7808904},
  url          = {https://doi.org/10.5281/zenodo.7808904}
}
```

Also consider starring [our github repo](https://github.com/avik-pal/Lux.jl/)

<p align="center">
    <img width="400px" src="assets/lux-logo.svg#gh-light-mode-only"/>
    <img width="400px" src="assets/lux-logo-dark.svg#gh-dark-mode-only"/>
</p>

<div align="center">

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/)

[![CI](https://github.com/LuxDL/Lux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/Lux.jl/actions/workflows/CI.yml)
[![Build status](https://img.shields.io/buildkite/ba1f9622add5978c2d7b194563fd9327113c9c21e5734be20e/main.svg?label=gpu&branch=main)](https://buildkite.com/julialang/lux-dot-jl)
[![codecov](https://codecov.io/gh/LuxDL/Lux.jl/branch/main/graph/badge.svg?token=IMqBM1e3hz)](https://codecov.io/gh/LuxDL/Lux.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Lux)](https://pkgs.genieframework.com?packages=Lux)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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

Look in the [examples](/examples/) directory for self-contained usage examples. The [documentation](https://lux.csail.mit.edu) has examples sorted into proper categories.

## Ecosystem

Checkout our [Ecosystem](http://lux.csail.mit.edu/dev/ecosystem/) page for more details. 

## Getting Help

For usage related questions, please use [Github Discussions](https://github.com/LuxDL/Lux.jl/discussions) or [JuliaLang Discourse (machine learning domain)](https://discourse.julialang.org/c/domain/ml/) which allows questions and answers to be indexed. To report bugs use [github issues](https://github.com/LuxDL/Lux.jl/issues) or even better send in a [pull request](https://github.com/LuxDL/Lux.jl/pulls).

## Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@software{pal2023lux,
  author    = {Pal, Avik},
  title     = {{Lux: Explicit Parameterization of Deep Neural Networks in Julia}},
  month     = apr,
  year      = 2023,
  note      = {If you use this software, please cite it as below.},
  publisher = {Zenodo},
  version   = {v0.5.0},
  doi       = {10.5281/zenodo.7808904},
  url       = {https://doi.org/10.5281/zenodo.7808904}
}

@thesis{pal2023efficient,
  title     = {{On Efficient Training \& Inference of Neural Differential Equations}},
  author    = {Pal, Avik},
  year      = {2023},
  school    = {Massachusetts Institute of Technology}
}
```

Also consider starring [our github repo](https://github.com/LuxDL/Lux.jl/)

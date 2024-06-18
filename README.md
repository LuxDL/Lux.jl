<p align="center">
    <img width="400px" src="assets/lux-logo.svg#gh-light-mode-only"/>
    <img width="400px" src="assets/lux-logo-dark.svg#gh-dark-mode-only"/>
</p>

<div align="center">

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Join the chat at https://julialang.org/slack #machine-learning](https://img.shields.io/badge/Slack-Slack?logo=slack&message=chat&color=4A154B&logoColor=4A154B&labelColor=389826)](https://julialang.org/slack/)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/)

[![CI](https://github.com/LuxDL/Lux.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LuxDL/Lux.jl/actions/workflows/CI.yml)
[![Build status](https://img.shields.io/buildkite/ba1f9622add5978c2d7b194563fd9327113c9c21e5734be20e/main.svg?label=gpu&branch=main)](https://buildkite.com/julialang/lux-dot-jl)
[![codecov](https://codecov.io/gh/LuxDL/Lux.jl/branch/main/graph/badge.svg?token=IMqBM1e3hz)](https://codecov.io/gh/LuxDL/Lux.jl)
[![Benchmarks](https://github.com/LuxDL/Lux.jl/actions/workflows/Benchmark.yml/badge.svg?branch=main)](https://lux.csail.mit.edu/benchmarks/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

</div>

The 🔥 Deep Learning Framework

## Installation

```julia
import Pkg
Pkg.add("Lux")
```

## Getting Started

```julia
using Lux, Random, Optimisers, Zygote
# using LuxCUDA, AMDGPU, Metal, oneAPI # Optional packages for GPU support

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

Checkout our [Ecosystem](http://lux.csail.mit.edu/dev/ecosystem) page for more details. 

## Testing

The full test of `Lux.jl` takes a long time, here's how to test a portion of the code.

For each `@testitem`, there are corresponding `tags`, for example:

```julia
@testitem "SkipConnection" setup=[SharedTestSetup] tags=[:core_layers]
```

For example, let's consider the tests for `SkipConnection`:

```julia
@testitem "SkipConnection" setup=[SharedTestSetup] tags=[:core_layers] begin
    .....
end
```

We can test the group to which `SkipConnection` belongs by testing `core_layers`.
To do so set the `LUX_TEST_GROUP` environment variable, or rename the tag to
further narrow the test scope:

```shell
export LUX_TEST_GROUP="core_layers"
```

Or directly modify the default test tag in `runtests.jl`:

```julia
# const LUX_TEST_GROUP = lowercase(get(ENV, "LUX_TEST_GROUP", "all"))
const LUX_TEST_GROUP = lowercase(get(ENV, "LUX_TEST_GROUP", "core_layers"))
```

But be sure to restore the default value "all" before submitting the code.

Furthermore if you want to run a specific test based on the name of the testset, you can
use [TestEnv.jl](https://github.com/JuliaTesting/TestEnv.jl) as follows. Start with activating the Lux environment and then run the following:

```julia
using TestEnv; TestEnv.activate(); using ReTestItems;

# Assuming you are in the main directory of Lux
ReTestItems.runtests("tests/"; name = <NAME OF THE TEST>)
```

For the `SkipConnection` tests that would be:

```julia
ReTestItems.runtests("tests/"; name = SkipConnection)
```

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

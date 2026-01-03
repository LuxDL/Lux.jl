<p align="center">
    <img width="400px" src="assets/lux-logo.svg#gh-light-mode-only"/>
    <img width="400px" src="assets/lux-logo-dark.svg#gh-dark-mode-only"/>
</p>

<div align="center">

[![GitHub Discussions](https://img.shields.io/github/discussions/LuxDL/Lux.jl?color=white&logo=github&label=Discussions)](https://github.com/LuxDL/Lux.jl/discussions)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/)

[![CI](https://github.com/LuxDL/Lux.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LuxDL/Lux.jl/actions/workflows/CI.yml)
[![CI (pre-release)](<https://img.shields.io/github/actions/workflow/status/LuxDL/Lux.jl/CIPreRelease.yml?branch=main&label=CI%20(pre-release)&logo=github>)](https://github.com/LuxDL/Lux.jl/actions/workflows/CIPreRelease.yml)
[![Build status](https://img.shields.io/buildkite/ba1f9622add5978c2d7b194563fd9327113c9c21e5734be20e/main.svg?label=gpu&branch=main&logo=buildkite)](https://buildkite.com/julialang/lux-dot-jl)
[![codecov](https://codecov.io/gh/LuxDL/Lux.jl/branch/main/graph/badge.svg?token=IMqBM1e3hz)](https://codecov.io/gh/LuxDL/Lux.jl)
<!-- [![Benchmarks](https://github.com/LuxDL/Lux.jl/actions/workflows/Benchmark.yml/badge.svg?branch=main)](https://lux.csail.mit.edu/benchmarks/) -->

[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FLux&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/Lux)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FLux&query=total_requests&&label=Total%20Downloads)](https://juliapkgstats.com/pkg/Lux)

[![JET Testing](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

</div>

<div align="center">
    <h2>Elegant & Performant Deep Learning in JuliaLang</h2>
    <h3>Model with the elegance of Julia, and the performance of XLA.</h3>
</div>

## 💻 Installation

```julia
import Pkg
Pkg.add("Lux")
```

> [!TIP]
> To use Lux online, use [Google Colab](https://colab.research.google.com/). The Julia Runtime comes pre-installed with Lux and Reactant!

<div align="center">

| **Packages**                                           | **Stable Version**                                             | **Monthly Downloads**                                                 | **Total Downloads**                                                         | **Build Status**                                                                                                                                |
| :----------------------------------------------------- | :------------------------------------------------------------- | :-------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| 📦 [Lux.jl](./src)                                     | [![][lux-version]][lux-juliahub]                               | [![][downloads-lux]][downloads-lux-url]                               | [![][total-downloads-lux]][downloads-lux-url]                               | [![][gh-actions-lux]][gh-actions-lux-url] [![][gh-actions-lux-prerelease]][gh-actions-lux-prerelease-url] [![][buildkite-badge]][buildkite-url] |
| └ 📦 [LuxLib.jl](./lib/LuxLib)                         | [![][luxlib-version]][luxlib-juliahub]                         | [![][downloads-luxlib]][downloads-luxlib-url]                         | [![][total-downloads-luxlib]][downloads-luxlib-url]                         | [![][gh-actions-luxlib]][gh-actions-luxlib-url]                                                                                                 |
| └ 📦 [LuxCore.jl](./lib/LuxCore)                       | [![][luxcore-version]][luxcore-juliahub]                       | [![][downloads-luxcore]][downloads-luxcore-url]                       | [![][total-downloads-luxcore]][downloads-luxcore-url]                       | [![][gh-actions-luxcore]][gh-actions-luxcore-url]                                                                                               |
| └ 📦 [MLDataDevices.jl](./lib/MLDataDevices)           | [![][mldatadevices-version]][mldatadevices-juliahub]           | [![][downloads-mldatadevices]][downloads-mldatadevices-url]           | [![][total-downloads-mldatadevices]][downloads-mldatadevices-url]           | [![][gh-actions-mldatadevices]][gh-actions-mldatadevices-url]                                                                                   |
| └ 📦 [WeightInitializers.jl](./lib/WeightInitializers) | [![][weightinitializers-version]][weightinitializers-juliahub] | [![][downloads-weightinitializers]][downloads-weightinitializers-url] | [![][total-downloads-weightinitializers]][downloads-weightinitializers-url] | [![][gh-actions-weightinitializers]][gh-actions-weightinitializers-url]                                                                         |
| └ 📦 [LuxTestUtils.jl](./lib/LuxTestUtils)             | [![][luxtestutils-version]][luxtestutils-juliahub]             | [![][downloads-luxtestutils]][downloads-luxtestutils-url]             | [![][total-downloads-luxtestutils]][downloads-luxtestutils-url]             | [![][gh-actions-luxtestutils]][gh-actions-luxtestutils-url]                                                                                     |
| └ 📦 [LuxCUDA.jl](./lib/LuxCUDA)                       | [![][luxcuda-version]][luxcuda-juliahub]                       | [![][downloads-luxcuda]][downloads-luxcuda-url]                       | [![][total-downloads-luxcuda]][downloads-luxcuda-url]                       | [![][gh-actions-luxcuda]][gh-actions-luxcuda-url]                                                                                               |

</div>

<!-- VARIABLES -->

<!-- Package -->

[lux-version]: https://juliahub.com/docs/General/Lux/stable/version.svg?color=blue
[luxlib-version]: https://juliahub.com/docs/General/LuxLib/stable/version.svg?color=blue
[luxcore-version]: https://juliahub.com/docs/General/LuxCore/stable/version.svg?color=blue
[mldatadevices-version]: https://juliahub.com/docs/General/MLDataDevices/stable/version.svg?color=blue
[weightinitializers-version]: https://juliahub.com/docs/General/WeightInitializers/stable/version.svg?color=blue
[luxtestutils-version]: https://juliahub.com/docs/General/LuxTestUtils/stable/version.svg?color=blue
[luxcuda-version]: https://juliahub.com/docs/General/LuxCUDA/stable/version.svg?color=blue
[lux-juliahub]: https://juliahub.com/ui/Packages/General/Lux
[luxlib-juliahub]: https://juliahub.com/ui/Packages/General/LuxLib
[luxcore-juliahub]: https://juliahub.com/ui/Packages/General/LuxCore
[mldatadevices-juliahub]: https://juliahub.com/ui/Packages/General/MLDataDevices
[weightinitializers-juliahub]: https://juliahub.com/ui/Packages/General/WeightInitializers
[luxtestutils-juliahub]: https://juliahub.com/ui/Packages/General/LuxTestUtils
[luxcuda-juliahub]: https://juliahub.com/ui/Packages/General/LuxCUDA

<!-- Documentation -->

[docr-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docd-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docr-url]: https://lux.csail.mit.edu/stable/
[docd-url]: https://lux.csail.mit.edu/dev/

<!-- Buildkite -->

[buildkite-badge]: https://img.shields.io/buildkite/ba1f9622add5978c2d7b194563fd9327113c9c21e5734be20e/main.svg?label=gpu&branch=main&logo=buildkite]

[buildkite-url]: https://buildkite.com/julialang/lux-dot-jl/builds?branch=main

<!-- CI -->

[gh-actions-lux]: https://github.com/LuxDL/Lux.jl/workflows/CI%20(Lux)/badge.svg
[gh-actions-lux-prerelease]: https://github.com/LuxDL/Lux.jl/workflows/CIPreRelease%20(Lux)/badge.svg
[gh-actions-luxlib]: https://github.com/LuxDL/Lux.jl/workflows/CI%20(LuxLib)/badge.svg
[gh-actions-luxcore]: https://github.com/LuxDL/Lux.jl/workflows/CI%20(LuxCore)/badge.svg
[gh-actions-mldatadevices]: https://github.com/LuxDL/Lux.jl/workflows/CI%20(MLDataDevices)/badge.svg
[gh-actions-weightinitializers]: https://github.com/LuxDL/Lux.jl/workflows/CI%20(WeightInitializers)/badge.svg
[gh-actions-luxtestutils]: https://github.com/LuxDL/Lux.jl/workflows/CI%20(LuxTestUtils)/badge.svg
[gh-actions-luxcuda]: https://github.com/LuxDL/Lux.jl/workflows/CI%20(LuxCUDA)/badge.svg
[gh-actions-lux-url]: https://github.com/LuxDL/Lux.jl/actions/workflows/CI.yml
[gh-actions-lux-prerelease-url]: https://github.com/LuxDL/Lux.jl/actions/workflows/CIPreRelease.yml
[gh-actions-luxlib-url]: https://github.com/LuxDL/Lux.jl/actions/workflows/CI_LuxLib.yml
[gh-actions-luxcore-url]: https://github.com/LuxDL/Lux.jl/actions/workflows/CI_LuxCore.yml
[gh-actions-mldatadevices-url]: https://github.com/LuxDL/Lux.jl/actions/workflows/CI_MLDataDevices.yml
[gh-actions-weightinitializers-url]: https://github.com/LuxDL/Lux.jl/actions/workflows/CI_WeightInitializers.yml
[gh-actions-luxtestutils-url]: https://github.com/LuxDL/Lux.jl/actions/workflows/CI_LuxTestUtils.yml
[gh-actions-luxcuda-url]: https://github.com/LuxDL/Lux.jl/actions/workflows/CI_LuxCUDA.yml

<!-- Downloads -->

[total-downloads-lux]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FLux&query=total_requests&label=Downloads
[total-downloads-luxlib]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FLuxLib&query=total_requests&label=Downloads
[total-downloads-luxcore]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FLuxCore&query=total_requests&label=Downloads
[total-downloads-mldatadevices]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FMLDataDevices&query=total_requests&label=Downloads
[total-downloads-weightinitializers]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FWeightInitializers&query=total_requests&label=Downloads
[total-downloads-luxtestutils]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FLuxTestUtils&query=total_requests&label=Downloads
[total-downloads-luxcuda]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FLuxCUDA&query=total_requests&label=Downloads
[downloads-lux]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FLux&query=total_requests&suffix=%2Fmonth&label=Downloads
[downloads-luxlib]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FLuxLib&query=total_requests&suffix=%2Fmonth&label=Downloads
[downloads-luxcore]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FLuxCore&query=total_requests&suffix=%2Fmonth&label=Downloads
[downloads-mldatadevices]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FMLDataDevices&query=total_requests&suffix=%2Fmonth&label=Downloads
[downloads-weightinitializers]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FWeightInitializers&query=total_requests&suffix=%2Fmonth&label=Downloads
[downloads-luxtestutils]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FLuxTestUtils&query=total_requests&suffix=%2Fmonth&label=Downloads
[downloads-luxcuda]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FLuxCUDA&query=total_requests&suffix=%2Fmonth&label=Downloads
[downloads-lux-url]: http://juliapkgstats.com/pkg/Lux
[downloads-luxlib-url]: http://juliapkgstats.com/pkg/LuxLib
[downloads-luxcore-url]: http://juliapkgstats.com/pkg/LuxCore
[downloads-mldatadevices-url]: http://juliapkgstats.com/pkg/MLDataDevices
[downloads-weightinitializers-url]: http://juliapkgstats.com/pkg/WeightInitializers
[downloads-luxtestutils-url]: http://juliapkgstats.com/pkg/LuxTestUtils
[downloads-luxcuda-url]: http://juliapkgstats.com/pkg/LuxCUDA

## 🚀 Benchmarks

Currently Benchmarks are scatter across a few places:

  1. For comparison with other Julia packages like CUDA.jl take a look
     at [Lux.jl/perf](./perf/README.md).
  2. <https://enzymead.github.io/Enzyme-JAX/benchmarks/> highlights
     performance of EnzymeJAX (backend for Reactant.jl) against JAX.
  3. <https://enzymead.github.io/Reactant.jl/benchmarks/> highlights
     performance of Reactant.jl against default XLA and base Julia
     compilation.

## 🤸 Quickstart

### Reactant & Enzyme

```julia
using Lux, Random, Optimisers, Reactant, Enzyme

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Chain(Dense(128, 256, tanh), Chain(Dense(256, 1, tanh), Dense(1, 10)))

dev = reactant_device()

ps, st = Lux.setup(rng, model) |> dev

x = rand(rng, Float32, 128, 2) |> dev

# We need to compile the model before we can use it.
model_forward = @compile model(x, ps, Lux.testmode(st))
model_forward(x, ps, Lux.testmode(st))

# Gradients can be computed using Enzyme
@jit Enzyme.gradient(Reverse, sum ∘ first ∘ Lux.apply, Const(model), x, ps, Const(st))

# All of this can be automated using the TrainState API
train_state = Training.TrainState(model, ps, st, Adam(0.001f0))

gs, loss, stats, train_state = Training.single_train_step!(
    AutoEnzyme(), MSELoss(),
    (x, dev(rand(rng, Float32, 10, 2))), train_state
)
```

### Native Julia & Zygote

```julia
using Lux, Random, Optimisers, Zygote
# using LuxCUDA, AMDGPU, Metal, oneAPI # Optional packages for GPU support

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

# Construct the layer
model = Chain(Dense(128, 256, tanh), Chain(Dense(256, 1, tanh), Dense(1, 10)))

# Get the device determined by Lux
dev = gpu_device()

# Parameter and State Variables
ps, st = Lux.setup(rng, model) |> dev

# Dummy Input
x = rand(rng, Float32, 128, 2) |> dev

# Run the model
y, st = Lux.apply(model, x, ps, st)

# Gradients
## First construct a TrainState
train_state = Lux.Training.TrainState(model, ps, st, Adam(0.0001f0))

## We can compute the gradients using Training.compute_gradients
gs, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), MSELoss(),
    (x, dev(rand(rng, Float32, 10, 2))), train_state)

## Optimization
train_state = Training.apply_gradients!(train_state, gs) # or Training.apply_gradients (no `!` at the end)

# Both these steps can be combined into a single call
gs, loss, stats, train_state = Training.single_train_step!(AutoZygote(), MSELoss(),
    (x, dev(rand(rng, Float32, 10, 2))), train_state)
```

## 📚 Examples

Look in the [examples](/examples/) directory for self-contained usage examples. The [documentation](https://lux.csail.mit.edu) has examples sorted into proper categories.

## 🆘 Getting Help

For usage related questions, please use [Github Discussions](https://github.com/orgs/LuxDL/discussions) which allows questions and answers to be indexed. To report bugs use [github issues](https://github.com/LuxDL/Lux.jl/issues) or even better send in a [pull request](https://github.com/LuxDL/Lux.jl/pulls).

## 🧑‍🔬 Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@software{pal2023lux,
  author    = {Pal, Avik},
  title     = {{Lux: Explicit Parameterization of Deep Neural Networks in Julia}},
  month     = apr,
  year      = 2023,
  note      = {If you use this software, please cite it as below.},
  publisher = {Zenodo},
  version   = {v1.4.2},
  doi       = {10.5281/zenodo.7808903},
  url       = {https://doi.org/10.5281/zenodo.7808903},
  swhid     = {swh:1:dir:1a304ec3243961314a1cc7c1481a31c4386c4a34;origin=https://doi.org/10.5281/zenodo.7808903;visit=swh:1:snp:e2bbe43b14bde47c4ddf7e637eb7fc7bd10db8c7;anchor=swh:1:rel:2c0c0ff927e7bfe8fc8bc43fd553ab392a6eb403;path=/}
}

@thesis{pal2023efficient,
  title     = {{On Efficient Training \& Inference of Neural Differential Equations}},
  author    = {Pal, Avik},
  year      = {2023},
  school    = {Massachusetts Institute of Technology}
}
```

Also consider starring [our github repo](https://github.com/LuxDL/Lux.jl/).

## 🧑‍💻 Contributing

This section is somewhat incomplete. You can contribute by contributing to finishing this
section 😜.

### 💎 Formatting (JuliaFormatter)

> [!NOTE]
> Pin JuliaFormatter to v1 until upstream issues with v2 are resolved.

```julia
using JuliaFormatter
format(".")
```

### 🧪 Testing

The full test of `Lux.jl` takes a long time, here's how to test a portion of the code.

For each `@testitem`, there are corresponding `tags`, for example:

```julia
@testitem "SkipConnection" setup=[SharedTestSetup] tags=[:core_layers]
```

For example, let's consider the tests for `SkipConnection`:

```julia
@testitem "SkipConnection" setup=[SharedTestSetup] tags=[:core_layers] begin
    ...
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
ReTestItems.runtests("tests/"; name = "NAME OF THE TEST")
```

For the `SkipConnection` tests that would be:

```julia
ReTestItems.runtests("tests/"; name = "SkipConnection")
```

### 📖 Documentation

Lux builds a bunch of tutorials as part of its documentation. This can be time-consuming and
requires a lot of compute. To speed up the build, you can set the
`LUX_DOCS_DRAFT_BUILD=true`.

```shell
LUX_DOCS_DRAFT_BUILD=true julia --threads=auto --startup=no --project=docs docs/make.jl
```

When writing tutorials (anything under `examples/`), include the tutorial in
`docs/tutorials.jl`. If the tutorial is time-consuming, set `should_run` to `false`.

Additionally for a new page to be included in the navigation and sidebar, these need to be
added to `docs/src/.vitepress/config.mts`. Specifically these need to be added under
`sidebar` and/or `nav` based on the type of page.

To use LiveServer to preview the docs locally, checkout
[DocumenterVitepress.jl](https://luxdl.github.io/DocumenterVitepress.jl/dev/manual/get_started#Preview-Documentation-Development-Instantly)
documentation.

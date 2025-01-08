```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: LuxDL Docs
  text: Elegant & Performant Scientific Machine Learning in JuliaLang
  tagline: A Pure Julia Deep Learning Framework designed for Scientific Machine Learning
  actions:
    - theme: brand
      text: Tutorials
      link: /tutorials
    - theme: alt
      text: API Reference 📚
      link: /api/Lux/layers
    - theme: alt
      text: View on GitHub
      link: https://github.com/LuxDL/Lux.jl
  image:
    src: /lux-logo.svg
    alt: Lux.jl

features:
  - icon: 🚀
    title: Fast & Extendable
    details: Lux.jl is written in Julia itself, making it extremely extendable. CUDA and AMDGPU are supported first-class, with experimental support for Metal and Intel GPUs.
    link: /introduction

  - icon: 🐎
    title: Powered by the XLA Compiler
    details: Lux.jl seamlessly integrates with Reactant.jl, to compiler models to run on CPU, GPU, TPU, and more.
    link: /manual/compiling_lux_models

  - icon: 🧑‍🔬
    title: SciML ❤️ Lux
    details: Lux is the default choice for all SciML packages, including DiffEqFlux.jl, NeuralPDE.jl, and more.
    link: https://sciml.ai/

  - icon: 🧩
    title: Uniquely Composable
    details: Lux.jl natively supports Arbitrary Parameter Types, making it uniquely composable with other Julia packages (and even Non-Julia packages).
    link: /api/Lux/contrib#Training
---
```

## How to Install Lux.jl?

Its easy to install Lux.jl. Since Lux.jl is registered in the Julia General registry,
you can simply run the following command in the Julia REPL:

```julia
julia> using Pkg
julia> Pkg.add("Lux")
```

If you want to use the latest unreleased version of Lux.jl, you can run the following
command: (in most cases the released version will be same as the version on github)

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/LuxDL/Lux.jl")
```

## Want GPU Support?

Install the following package(s):

:::code-group

```julia [NVIDIA GPUs]
using Pkg
Pkg.add("LuxCUDA")
# or
Pkg.add(["CUDA", "cuDNN"])
```

```julia [AMD ROCm GPUs]
using Pkg
Pkg.add("AMDGPU")
```

```julia [Metal M-Series GPUs]
using Pkg
Pkg.add("Metal")
```

```julia [Intel GPUs]
using Pkg
Pkg.add("oneAPI")
```

:::

Run the following to access a device:

:::code-group

```julia [NVIDIA GPUs]
using Lux, LuxCUDA

const dev = gpu_device()
```

```julia [AMD ROCm GPUs]
using Lux, AMDGPU

const dev = gpu_device()
```

```julia [Metal M-Series GPUs]
using Lux, Metal

const dev = gpu_device()
```

```julia [Intel GPUs]
using Lux, oneAPI

const dev = gpu_device()
```

:::

## Want Reactant (XLA) Support?

Install the following package:

```julia
using Pkg;
Pkg.add("Reactant")
```

Run the following to access a device (Reactant automatically selects the best backend by
default):

:::code-group

```julia [CPU Backend]
using Reactant, Lux
Reactant.set_default_backend("cpu")

const dev = reactant_device()
```

```julia [GPU Backend]
using Reactant, Lux
Reactant.set_default_backend("gpu")

const dev = reactant_device()
```

```julia [TPU Backend]
using Reactant, Lux
Reactant.set_default_backend("tpu")

const dev = reactant_device()
```

:::

```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: LuxDL Docs
  text: Elegant & Performant Deep Learning in JuliaLang
  tagline: Model with the elegance of Julia, and the performance of XLA.
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
    details: Lux.jl seamlessly integrates with Reactant.jl, to compile models to run on CPU, GPU, TPU, and more.
    link: /manual/compiling_lux_models

  - icon: 🧑‍🔬
    title: SciML ❤️ Lux
    details: Lux is the default choice for all SciML packages, including DiffEqFlux.jl, NeuralPDE.jl, and more.
    link: https://sciml.ai/

  - icon: 🏭
    title: Production-Ready
    details: Seamlessly deploy models to servers and edge devices by leveraging the rich tensorflow ecosystem.
    link: /api/Lux/serialization
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

```julia [GPU Backend (CUDA / ROCM)]
using Reactant, Lux
Reactant.set_default_backend("gpu")

const dev = reactant_device()
```

```julia [TPU Backend]
using Reactant, Lux
Reactant.set_default_backend("tpu")

const dev = reactant_device()
```

```julia [Tenstorrent Backend (Experimental)]
using Reactant, Lux
Reactant.set_default_backend("tt")

const dev = reactant_device()
```

:::

## Want GPU Support?

!!! tip "Prefer using Reactant"

    In almost all cases, we recommend using Reactant.jl for GPU support.

!!! warning "AMD GPU Support"

    For AMD GPUs, we **strongly recommend using Reactant** instead of AMDGPU.jl directly.
    Native AMDGPU.jl support is experimental and has known limitations including deadlocks
    in certain situations. Reactant provides better performance and reliability for AMD GPUs.

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

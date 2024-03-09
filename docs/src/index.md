```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: LuxDL Docs
  text: Elegant & Performant Deep Learning in JuliaLang
  tagline: A Pure Julia Deep Learning Framework putting Correctness and Performance First
  actions:
    - theme: brand
      text: Tutorials
      link: /dev/tutorials/
    - theme: alt
      text: Ecosystem
      link: /dev/ecosystem
    - theme: alt
      text: API Reference 📚
      link: /dev/api/Lux/layers
    - theme: alt
      text: View on GitHub
      link: https://github.com/LuxDL/Lux.jl
  image:
    src: /lux-logo.svg
    alt: Lux.jl

features:
  - icon: 🚀
    title: Fast & Extendible
    details: Lux.jl is written in Julia itself, making it extremely extendible. <u><a href="https://github.com/JuliaGPU/CUDA.jl">CUDA</a></u> and <u><a href="https://github.com/JuliaGPU/AMDGPU.jl">AMDGPU</a></u> are supported first-class, with experimental support for <u><a href="https://github.com/JuliaGPU/Metal.jl">Metal</a></u> Hardware.

  - icon: 🧑‍🔬
    title: SciML ❤️ Lux
    details: Lux is the default choice for many <u><a href="https://github.com/SciML">SciML</a></u> packages, including DiffEqFlux.jl, NeuralPDE.jl, and more.

  - icon: 🧩
    title: Uniquely Composable
    details: Lux.jl natively supports Arbitrary Parameter Types, making it uniquely composable with other Julia packages (and even Non-Julia packages).

  - icon: 🧪
    title: Well Tested
    details: Lux.jl tests every supported Automatic Differentiation Framework with every supported hardware backend against Finite Differences to prevent sneaky 🐛 in your code.
---
```

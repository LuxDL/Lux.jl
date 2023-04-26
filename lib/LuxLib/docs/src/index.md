# LuxLib

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://luxdl.github.io/LuxLib.jl/dev)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://luxdl.github.io/LuxLib.jl/stable)

[![CI](https://github.com/LuxDL/LuxLib.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/LuxLib.jl/actions/workflows/CI.yml)
[![Buildkite](https://img.shields.io/buildkite/650bceb9ffcb044bee9c21e591728aaac2d8b57fae466e99cd/main?label=gpu)](https://buildkite.com/julialang/luxlib-dot-jl)
[![codecov](https://codecov.io/gh/LuxDL/LuxLib.jl/branch/main/graph/badge.svg?token=1ZY0A2NPEM)](https://codecov.io/gh/LuxDL/LuxLib.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/LuxLib)](https://pkgs.genieframework.com?packages=LuxLib)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Backend for [Lux.jl](http://lux.csail.mit.edu/stable).

```@meta
CurrentModule = LuxLib
```

## API Reference

### Index

```@index
Pages = ["index.md"]
```

### Dropout

```@docs
alpha_dropout
dropout
```

### Normalization

```@docs
batchnorm
groupnorm
instancenorm
layernorm
```

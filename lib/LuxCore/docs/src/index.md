# LuxCore

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://luxdl.github.io/LuxCore.jl/dev)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://luxdl.github.io/LuxCore.jl/stable)

[![CI](https://github.com/LuxDL/LuxCore.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/LuxCore.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/LuxDL/LuxCore.jl/branch/main/graph/badge.svg?token=1ZY0A2NPEM)](https://codecov.io/gh/LuxDL/LuxCore.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/LuxCore)](https://pkgs.genieframework.com?packages=LuxCore)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

`LuxCore.jl` defines the abstract layers for Lux. Allows users to be compatible with the
entirely of `Lux.jl` without having such a heavy dependency. If you are depending on
`Lux.jl` directly, you do not need to depend on `LuxCore.jl` (all the functionality is
exported via `Lux.jl`).

```@meta
CurrentModule = LuxCore
```

## API Reference

### Index

```@index
Pages = ["index.md"]
```

### Abstract Types

```@docs
LuxCore.AbstractExplicitLayer
LuxCore.AbstractExplicitContainerLayer
```

### General

```@docs
LuxCore.apply
LuxCore.setup
```

### Parameters

```@docs
LuxCore.initialparameters
LuxCore.parameterlength
```

### States

```@docs
LuxCore.initialstates
LuxCore.statelength
LuxCore.testmode
LuxCore.trainmode
LuxCore.update_state
```

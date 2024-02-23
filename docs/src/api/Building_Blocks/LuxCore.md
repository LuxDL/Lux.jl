```@meta
CurrentModule = Lux
```

# LuxCore

`LuxCore.jl` defines the abstract layers for Lux. Allows users to be compatible with the
entirely of `Lux.jl` without having such a heavy dependency. If you are depending on
`Lux.jl` directly, you do not need to depend on `LuxCore.jl` (all the functionality is
exported via `Lux.jl`).

## Index

```@index
Pages = ["LuxCore.md"]
```

## Abstract Types

```@docs
LuxCore.AbstractExplicitLayer
LuxCore.AbstractExplicitContainerLayer
```

## General

```@docs
LuxCore.apply
LuxCore.stateless_apply
LuxCore.check_fmap_condition
LuxCore.contains_lux_layer
LuxCore.display_name
LuxCore.replicate
LuxCore.setup
```

## Parameters

```@docs
LuxCore.initialparameters
LuxCore.parameterlength
```

## States

```@docs
LuxCore.initialstates
LuxCore.statelength
LuxCore.testmode
LuxCore.trainmode
LuxCore.update_state
```

## Layer size

::: warning

These specifications have been added very recently and most layers currently do not
implement them.

:::

```@docs
LuxCore.inputsize
LuxCore.outputsize
```

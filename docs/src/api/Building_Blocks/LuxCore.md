```@meta
CollapsedDocStrings = true
```

# LuxCore

`LuxCore.jl` defines the abstract layers for Lux. Allows users to be compatible with the
entirely of `Lux.jl` without having such a heavy dependency. If you are depending on
`Lux.jl` directly, you do not need to depend on `LuxCore.jl` (all the functionality is
exported via `Lux.jl`).

## Abstract Types

```@docs
LuxCore.AbstractLuxLayer
LuxCore.AbstractLuxWrapperLayer
LuxCore.AbstractLuxContainerLayer
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
LuxCore.preserves_state_type
```

## Layer size

```@docs
LuxCore.outputsize
```

## Stateful Layer

```@docs
LuxCore.StatefulLuxLayer
```

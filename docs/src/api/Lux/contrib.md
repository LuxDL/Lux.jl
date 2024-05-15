# Experimental Features

```@meta
CurrentModule = Lux
```

All features listed on this page are **experimental** which means:

1. No SemVer Guarantees. We use code here to iterate fast. That said, historically we have
   never broken any code in this module and have always provided a deprecation period.
2. Expect edge-cases and report them. It will help us move these features out of
   experimental sooner.
3. None of the features are exported.

!!! warning

    Starting v"0.5.2" all Experimental features need to be accessed via
    `Lux.Experimental.<feature>`. Direct access via `Lux.<feature>` will be removed in
    v"0.6".

## Index

```@index
Pages = ["contrib.md"]
```

## Training

Helper Functions making it easier to train `Lux.jl` models.

Lux.Training is meant to be simple and provide extremely basic functionality. We provide
basic building blocks which can be seamlessly composed to create complex training pipelines.

```@docs
Lux.Experimental.TrainState
Lux.Experimental.compute_gradients
Lux.Experimental.apply_gradients
Lux.Experimental.apply_gradients!
```

## Parameter Freezing

!!! info

    In the long term, this will be supported via
    [Optimisers.jl](https://github.com/FluxML/Optimisers.jl/pull/49).

```@docs
Lux.Experimental.FrozenLayer
Lux.Experimental.freeze
Lux.Experimental.unfreeze
```

For detailed usage example look at the [manual page](@ref freezing-model-parameters).

## Map over Layer

```@docs
Lux.Experimental.layer_map
Lux.Experimental.@layer_map
```

## Debugging Functionality

Model not working properly! Here are some functionalities to help you debug you Lux model.

```@docs
Lux.Experimental.@debug_mode
Lux.Experimental.DebugLayer
```

## Tied Parameters

```@docs
Lux.Experimental.share_parameters
```

## StatefulLuxLayer

[`Lux.StatefulLuxLayer`](@ref) used to be part of experimental features, but has been
promoted to stable API. It is now available via `Lux.StatefulLuxLayer`. Change all uses of
`Lux.Experimental.StatefulLuxLayer` to `Lux.StatefulLuxLayer`.

## Compact Layer API

[`Lux.@compact`](@ref) used to be part of experimental features, but has been promoted to
stable API. It is now available via `Lux.@compact`. Change all uses of
`Lux.Experimental.@compact` to `Lux.@compact`.

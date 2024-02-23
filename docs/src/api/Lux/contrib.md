# Experimental Features

```@meta
CurrentModule = Lux
```

All features listed on this page are **experimental** which means:

1. No SemVer Guarantees. We use code here to iterate fast and most users should wait for
   these features to be marked non-experimental.
2. The code will probably be moved into a separate repository in the future.
3. Expect edge-cases and report them. It will help us move these features out of
   experimental sooner.
4. None of the features are exported.

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

## Compact Layer API

```@docs
Lux.Experimental.@compact
```

```@meta
CollapsedDocStrings = true
```

# Experimental Features

All features listed on this page are **experimental** which means:

1. No SemVer Guarantees. We use code here to iterate fast. That said, historically we have
   never broken any code in this module and have always provided a deprecation period.
2. Expect edge-cases and report them. It will help us move these features out of
   experimental sooner.
3. None of the features are exported.

## Parameter Freezing

```@docs
Lux.Experimental.FrozenLayer
Lux.Experimental.freeze
Lux.Experimental.unfreeze
```

For detailed usage example look at the [manual page](@ref freezing-model-parameters).

## Map over Layer

```@docs
Lux.Experimental.layer_map
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

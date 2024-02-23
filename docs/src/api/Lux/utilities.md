# Utilities

```@meta
CurrentModule = Lux
```

## Index

```@index
Pages = ["utilities.md"]
```

## Device Management / Data Transfer

```@docs
Lux.cpu
Lux.gpu
```

!!! warning

    For detailed API documentation on Data Transfer check out the
    [LuxDeviceUtils.jl](@ref LuxDeviceUtils-API)

## Weight Initialization

!!! warning

    For API documentation on Initialization check out the
    [WeightInitializers.jl](@ref WeightInitializers-API)

## Miscellaneous Utilities

```@docs
Lux.foldl_init
Lux.istraining
Lux.multigate
```

## Updating Floating Point Precision

By default, Lux uses Float32 for all parameters and states. To update the precision
simply pass the parameters / states / arrays into one of the following functions.

```@docs
Lux.f16
Lux.f32
Lux.f64
```

## Stateful Layer

```@docs
StatefulLuxLayer
```


## Truncated Stacktraces

```@docs
Lux.disable_stacktrace_truncation!
```

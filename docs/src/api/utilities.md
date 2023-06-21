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
Lux.cpu_device
Lux.gpu_device
Lux.supported_gpu_backends
Lux.cpu
Lux.gpu
```

## Initialization

```@docs
Lux.glorot_normal
Lux.glorot_uniform
Lux.kaiming_normal
Lux.kaiming_uniform
Lux.ones32
Lux.rand32
Lux.randn32
Lux.zeros32
```

!!! note
    For detailed API documentation on Initialization check out the [WeightInitializers.jl](https://luxdl.github.io/WeightInitializers.jl/dev)

## Miscellaneous Utilities

```@docs
Lux.foldl_init
Lux.istraining
Lux.multigate
Lux.replicate
```

## Truncated Stacktraces

```@docs
Lux.disable_stacktrace_truncation!
```

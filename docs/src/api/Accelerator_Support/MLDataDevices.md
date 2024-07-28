# [MLDataDevices](@id MLDataDevices-API)

`MLDataDevices.jl` is a lightweight package defining rules for transferring data across
devices. Most users should directly use Lux.jl instead.

## Index

```@index
Pages = ["MLDataDevices.md"]
```

## Preferences

```@docs
gpu_backend!
```

## Data Transfer

```@docs
cpu_device
gpu_device
```

## Miscellaneous

```@docs
reset_gpu_device!
supported_gpu_backends
default_device_rng
get_device
get_device_type
MLDataDevices.loaded
MLDataDevices.functional
```

## Multi-GPU Support

```@docs
MLDataDevices.set_device!
```

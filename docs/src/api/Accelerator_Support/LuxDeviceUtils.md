# [LuxDeviceUtils](@id LuxDeviceUtils-API)

`LuxDeviceUtils.jl` is a lightweight package defining rules for transferring data across
devices. Most users should directly use Lux.jl instead.

## Index

```@index
Pages = ["LuxDeviceUtils.md"]
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
LuxDeviceUtils.loaded
LuxDeviceUtils.functional
```

## Multi-GPU Support

```@docs
LuxDeviceUtils.set_device!
```

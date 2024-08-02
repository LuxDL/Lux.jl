# [LuxDeviceUtils](@id LuxDeviceUtils-API)

`LuxDeviceUtils.jl` is a lightweight package defining rules for transferring data across
devices. Most users should directly use Lux.jl instead.

!!! note "Transition to `MLDataDevices.jl`"

    Currently this package is in maintenance mode and won't receive any new features,
    however, we will backport bug fixes till Lux `v1.0` is released. Post that this package
    should be considered deprecated and users should switch to `MLDataDevices.jl`.

    For more information on `MLDataDevices.jl` checkout the
    [MLDataDevices.jl Documentation](@ref MLDataDevices-API).

## Index

```@index
Pages = ["LuxDeviceUtils.md"]
```

## Preferences

```@docs
LuxDeviceUtils.gpu_backend!
```

## Data Transfer

```@docs
LuxDeviceUtils.cpu_device
LuxDeviceUtils.gpu_device
```

## Miscellaneous

```@docs
LuxDeviceUtils.reset_gpu_device!
LuxDeviceUtils.supported_gpu_backends
LuxDeviceUtils.default_device_rng
LuxDeviceUtils.get_device
LuxDeviceUtils.get_device_type
LuxDeviceUtils.loaded
LuxDeviceUtils.functional
```

## Multi-GPU Support

```@docs
LuxDeviceUtils.set_device!
```

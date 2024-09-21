```@meta
CollapsedDocStrings = true
```

# [MLDataDevices](@id MLDataDevices-API)

`MLDataDevices.jl` is a lightweight package defining rules for transferring data across
devices. Most users should directly use Lux.jl instead.

!!! note "Transitioning from `LuxDeviceUtils.jl`"

    `LuxDeviceUtils.jl` was renamed to `MLDataDevices.jl` in v1.0 as a part of allowing
    these packages to have broader adoption outsize the Lux community. However, Lux
    currently still uses `LuxDeviceUtils.jl` internally. This is supposed to change with
    the transition of Lux to `v1.0`.

## Preferences

```@docs
MLDataDevices.gpu_backend!
```

## Data Transfer

```@docs
MLDataDevices.cpu_device
MLDataDevices.gpu_device
```

## Miscellaneous

```@docs
MLDataDevices.reset_gpu_device!
MLDataDevices.supported_gpu_backends
MLDataDevices.default_device_rng
MLDataDevices.get_device
MLDataDevices.get_device_type
MLDataDevices.loaded
MLDataDevices.functional
```

## Multi-GPU Support

```@docs
MLDataDevices.set_device!
```

## Iteration

```@docs
MLDataDevices.DeviceIterator
```

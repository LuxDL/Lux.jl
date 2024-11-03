```@meta
CollapsedDocStrings = true
```

# [MLDataDevices](@id MLDataDevices-API)

`MLDataDevices.jl` is a lightweight package defining rules for transferring data across
devices.

## Preferences

```@docs
MLDataDevices.gpu_backend!
```

## Data Transfer

```@docs
MLDataDevices.cpu_device
MLDataDevices.gpu_device
MLDataDevices.reactant_device
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
MLDataDevices.isleaf
```

## Multi-GPU Support

```@docs
MLDataDevices.set_device!
```

## Iteration

```@docs
MLDataDevices.DeviceIterator
```

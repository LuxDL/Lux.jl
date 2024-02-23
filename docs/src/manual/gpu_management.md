# GPU Management

!!! info

    Starting from `v0.5`, Lux has transitioned to a new GPU management system. The old
    system using `cpu` and `gpu` functions is still in place but will be removed in `v0.6`.
    Using the  old functions might lead to performance regressions if used inside
    performance critical code.

`Lux.jl` can handle multiple GPU backends. Currently, the following backends are supported:

```@example gpu_management
using Lux, LuxCUDA, LuxAMDGPU  # Important to load trigger packages

supported_gpu_backends()
```

!!! danger "Metal Support"

    Support for Metal GPUs should be considered extremely experimental at this point.

## Automatic Backend Management (Recommended Approach)

Automatic Backend Management is done by two simple functions: `cpu_device` and `gpu_device`.

1. [`cpu_device`](@ref): This is a simple function and just returns a `LuxCPUDevice` object.

```@example gpu_management
cdev = cpu_device()
```

```@example gpu_management
x_cpu = randn(Float32, 3, 2)
```

2. [`gpu_device`](@ref): This function performs automatic GPU device selection and returns
   an object.
   1. If no GPU is available, it returns a `LuxCPUDevice` object.
   2. If a LocalPreferences file is present, then the backend specified in the file is used.
      To set a backend, use `Lux.gpu_backend!(<backend_name>)`.
      1. If the trigger package corresponding to the device is not loaded, then a warning is
         displayed.
      2. If no LocalPreferences file is present, then the first working GPU with loaded
         trigger package is used.


```@example gpu_management
gdev = gpu_device()

x_gpu = x_cpu |> gdev
```

```@example gpu_management
(x_gpu |> cdev) ≈ x_cpu
```

## Manual Backend Management

Automatic Device Selection can be circumvented by directly using `LuxCPUDevice` and
`AbstractLuxGPUDevice` objects.

```@example gpu_management
cdev = LuxCPUDevice()

x_cpu = randn(Float32, 3, 2)

if LuxCUDA.functional()
    gdev = LuxCUDADevice()
    x_gpu = x_cpu |> gdev
elseif LuxAMDGPU.functional()
    gdev = LuxAMDGPUDevice()
    x_gpu = x_cpu |> gdev
else
    @info "No GPU is available. Using CPU."
    x_gpu = x_cpu
end

(x_gpu |> cdev) ≈ x_cpu
```

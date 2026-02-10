---
url: /dev/api/Accelerator_Support/MLDataDevices.md
---
# MLDataDevices {#MLDataDevices-API}

`MLDataDevices.jl` is a lightweight package defining rules for transferring data across devices.

## Preferences {#Preferences}

```julia
gpu_backend!() = gpu_backend!("")
gpu_backend!(backend) = gpu_backend!(string(backend))
gpu_backend!(backend::AbstractGPUDevice)
gpu_backend!(backend::String)
```

Creates a `LocalPreferences.toml` file with the desired GPU backend.

If `backend == ""`, then the `gpu_backend` preference is deleted. Otherwise, `backend` is validated to be one of the possible backends and the preference is set to `backend`.

If a new backend is successfully set, then the Julia session must be restarted for the change to take effect.

source

## Data Transfer {#Data-Transfer}

```julia
cpu_device(eltype=missing) -> CPUDevice
```

Return a `CPUDevice` object which can be used to transfer data to CPU.

The `eltype` parameter controls element type conversion:

* `missing/nothing` (default): Preserves the original element type

* `Type{<:AbstractFloat}`: Converts floating-point arrays to the specified type

source

```julia
gpu_device(
    eltype::Union{Missing, Nothing, Type{<:AbstractFloat}}=missing;
    kwargs...
) -> AbstractDevice
gpu_device(
    device_id::Union{Nothing, Integer}=nothing
    eltype::Union{Missing, Nothing, Type{<:AbstractFloat}}=missing;
    force::Bool=false
) -> AbstractDevice
```

Selects GPU device based on the following criteria:

1. If `gpu_backend` preference is set and the backend is functional on the system, then that device is selected.

2. Otherwise, an automatic selection algorithm is used. We go over possible device backends in the order specified by `supported_gpu_backends()` and select the first functional backend.

3. If no GPU device is functional and  `force` is `false`, then `cpu_device()` is invoked.

4. If nothing works, an error is thrown.

**Arguments**

* `device_id::Union{Nothing, Integer}`: The device id to select. If `nothing`, then we return the last selected device or if none was selected then we run the autoselection and choose the current device using `CUDA.device()` or `AMDGPU.device()` or similar. If `Integer`, then we select the device with the given id. Note that this is `1`-indexed, in contrast to the `0`-indexed `CUDA.jl`. For example, `id = 4` corresponds to `CUDA.device!(3)`.

* `eltype::Union{Missing, Nothing, Type{<:AbstractFloat}}`: The element type to use for the device.
  * `missing` (default): Device specific. For `CUDADevice` this calls `CUDA.cu(x)`, for `AMDGPUDevice` this calls `AMDGPU.roc(x)`, for `MetalDevice` this calls `Metal.mtl(x)`, for `oneAPIDevice` this calls `oneArray(x)`, for `OpenCLDevice` this calls `CLArray(x)`.

  * `nothing`: Preserves the original element type.

  * `Type{<:AbstractFloat}`: Converts floating-point arrays to the specified type.

::: warning Warning

`device_id` is only applicable for `CUDA` and `AMDGPU` backends. For `Metal`, `oneAPI`, `OpenCL` and `CPU` backends, `device_id` is ignored and a warning is printed.

:::

::: warning Warning

`gpu_device` won't select a CUDA device unless both CUDA.jl and cuDNN.jl are loaded. This is to ensure that deep learning operations work correctly. Nonetheless, if cuDNN is not loaded you can still manually create a `CUDADevice` object and use it (e.g. `dev = CUDADevice()`).

:::

**Keyword Arguments**

* `force::Bool`: If `true`, then an error is thrown if no functional GPU device is found.

source

```julia
reactant_device(;
    force::Bool=false, client=missing, device=missing, sharding=missing, eltype=missing,
    track_numbers::Type{TN}=Union{}
) -> Union{ReactantDevice, CPUDevice}
```

Return a `ReactantDevice` object if functional. Otherwise, throw an error if `force` is `true`. Falls back to `CPUDevice` if `force` is `false`.

`client` and `device` are used to specify the client and particular device to use. If not specified, then the default client and index are used.

`sharding` is used to specify the sharding strategy. If a `Reactant.Sharding.AbstractSharding` is specified, then we use it to shard all abstract arrays. Alternatively, pass in a `IdDict` to specify the sharding for specific leaves.

`track_numbers` can be specified to convert numbers of specified subtypes to be traced.

The `eltype` parameter controls element type conversion:

* `missing/nothing` (default): Preserves the original element type

* `Type{<:AbstractFloat}`: Converts floating-point arrays to the specified type

source

## Miscellaneous {#Miscellaneous}

```julia
reset_gpu_device!()
```

Resets the selected GPU device. This is useful when automatic GPU selection needs to be run again.

source

```julia
supported_gpu_backends() -> Tuple{String, ...}
```

Return a tuple of supported GPU backends.

::: warning Warning

This is not the list of functional backends on the system, but rather backends which `MLDataDevices.jl` supports.

:::

source

```julia
default_device_rng(::AbstractDevice)
```

Returns the default RNG for the device. This can be used to directly generate parameters and states on the device using [WeightInitializers.jl](https://github.com/LuxDL/WeightInitializers.jl).

source

```julia
get_device(x) -> dev::AbstractDevice | Exception | Nothing
```

If all arrays (on the leaves of the structure) are on the same device, we return that device. Otherwise, we throw an error. If the object is device agnostic, we return `nothing`.

::: tip Note

Trigger Packages must be loaded for this to return the correct device.

:::

**Special Retuened Values**

* `nothing` – denotes that the object is device agnostic. For example, scalar, abstract range, etc.

* `UnknownDevice()` – denotes that the device type is unknown.

See also [`get_device_type`](/api/Accelerator_Support/MLDataDevices#MLDataDevices.get_device_type) for a faster alternative that can be used for dispatch based on device type.

source

```julia
get_device_type(x) -> Type{<:AbstractDevice} | Exception | Type{Nothing}
```

Similar to [`get_device`](/api/Accelerator_Support/MLDataDevices#MLDataDevices.get_device) but returns the type of the device instead of the device itself. This value is often a compile time constant and is recommended to be used instead of [`get_device`](/api/Accelerator_Support/MLDataDevices#MLDataDevices.get_device) where ever defining dispatches based on the device type.

::: tip Note

Trigger Packages must be loaded for this to return the correct device.

:::

**Special Retuened Values**

* `Nothing` – denotes that the object is device agnostic. For example, scalar, abstract range, etc.

* `UnknownDevice` – denotes that the device type is unknown.

source

```julia
loaded(x::AbstractDevice) -> Bool
loaded(::Type{<:AbstractDevice}) -> Bool
```

Checks if the trigger package for the device is loaded. Trigger packages are as follows:

* `CUDA.jl` and `cuDNN.jl` (or just `LuxCUDA.jl`) for NVIDIA CUDA Support.

* `AMDGPU.jl` for AMD GPU ROCM Support.

* `Metal.jl` for Apple Metal GPU Support.

* `oneAPI.jl` for Intel oneAPI GPU Support.

* `OpenCL.jl` for OpenCL support.

source

```julia
functional(x::AbstractDevice) -> Bool
functional(::Type{<:AbstractDevice}) -> Bool
```

Checks if the device is functional. This is used to determine if the device can be used for computation. Note that even if the backend is loaded (as checked via [`MLDataDevices.loaded`](/api/Accelerator_Support/MLDataDevices#MLDataDevices.loaded)), the device may not be functional.

Note that while this function is not exported, it is considered part of the public API.

source

```julia
isleaf(x) -> Bool
```

Returns `true` if `x` is a leaf node in the data structure.

Defining `MLDataDevices.isleaf(x::T) = true` for custom types can be used to customize the behavior the data movement behavior when an object with nested structure containing the type is transferred to a device.

`Adapt.adapt_structure(::AbstractDevice, x::T)` or `Adapt.adapt_structure(::AbstractDevice, x::T)` will be called during data movement if `isleaf(x::T)`.

If `MLDataDevices.isleaf(x::T)` is not defined, then it will fall back to `Functors.isleaf(x)`.

source

## Multi-GPU Support {#Multi-GPU-Support}

```julia
set_device!(T::Type{<:AbstractDevice}, dev_or_id)
```

Set the device for the given type. This is a no-op for `CPUDevice`. For `CUDADevice` and `AMDGPUDevice`, it prints a warning if the corresponding trigger package is not loaded.

Currently, `MetalDevice` and `oneAPIDevice` don't support setting the device.

**Arguments**

* `T::Type{<:AbstractDevice}`: The device type to set.

* `dev_or_id`: Can be the device from the corresponding package. For example for CUDA it can be a `CuDevice`. If it is an integer, it is the device id to set. This is `1`-indexed.

::: danger Danger

This specific function should be considered experimental at this point and is currently provided to support distributed training in Lux. As such please use `Lux.DistributedUtils` instead of using this function.

:::

source

```julia
set_device!(T::Type{<:AbstractDevice}, ::Nothing, rank::Integer)
```

Set the device for the given type. This is a no-op for `CPUDevice`. For `CUDADevice` and `AMDGPUDevice`, it prints a warning if the corresponding trigger package is not loaded.

Currently, `MetalDevice` and `oneAPIDevice` don't support setting the device.

**Arguments**

* `T::Type{<:AbstractDevice}`: The device type to set.

* `rank::Integer`: Local Rank of the process. This is applicable for distributed training and must be `0`-indexed.

::: danger Danger

This specific function should be considered experimental at this point and is currently provided to support distributed training in Lux. As such please use `Lux.DistributedUtils` instead of using this function.

:::

source

## Iteration {#Iteration}

```julia
DeviceIterator(dev::AbstractDevice, iterator)
```

Create a `DeviceIterator` that iterates through the provided `iterator` via `iterate`. Upon each iteration, the current batch is copied to the device `dev`, and the previous iteration is marked as freeable from GPU memory (via `unsafe_free!`) (no-op for a CPU device).

The conversion follows the same semantics as `dev(<item from iterator>)`.

::: tip Similarity to `CUDA.CuIterator`

The design inspiration was taken from `CUDA.CuIterator` and was generalized to work with other backends and more complex iterators (using `Functors`).

:::

::: tip `MLUtils.DataLoader`

Calling `dev(::MLUtils.DataLoader)` will automatically convert the dataloader to use the same semantics as `DeviceIterator`. This is generally preferred over looping over the dataloader directly and transferring the data to the device.

:::

**Examples**

The following was run on a computer with an NVIDIA GPU.

```julia
julia> using MLDataDevices, MLUtils

julia> X = rand(Float64, 3, 33);

julia> dataloader = DataLoader(X; batchsize=13, shuffle=false);

julia> for (i, x) in enumerate(dataloader)
           @show i, summary(x)
       end
(i, summary(x)) = (1, "3×13 Matrix{Float64}")
(i, summary(x)) = (2, "3×13 Matrix{Float64}")
(i, summary(x)) = (3, "3×7 Matrix{Float64}")

julia> for (i, x) in enumerate(CUDADevice()(dataloader))
           @show i, summary(x)
       end
(i, summary(x)) = (1, "3×13 CuArray{Float32, 2, CUDA.DeviceMemory}")
(i, summary(x)) = (2, "3×13 CuArray{Float32, 2, CUDA.DeviceMemory}")
(i, summary(x)) = (3, "3×7 CuArray{Float32, 2, CUDA.DeviceMemory}")
```

source

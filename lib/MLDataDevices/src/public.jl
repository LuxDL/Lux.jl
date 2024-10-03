struct CPUDevice <: AbstractCPUDevice end

@kwdef struct CUDADevice{D} <: AbstractGPUDevice
    device::D = nothing
end
@kwdef struct AMDGPUDevice{D} <: AbstractGPUDevice
    device::D = nothing
end
struct MetalDevice <: AbstractGPUDevice end
struct oneAPIDevice <: AbstractGPUDevice end

# TODO: Later we might want to add the client field here?
struct XLADevice <: AbstractAcceleratorDevice end

"""
    functional(x::AbstractDevice) -> Bool
    functional(::Type{<:AbstractDevice}) -> Bool

Checks if the device is functional. This is used to determine if the device can be used for
computation. Note that even if the backend is loaded (as checked via
[`MLDataDevices.loaded`](@ref)), the device may not be functional.

Note that while this function is not exported, it is considered part of the public API.
"""
functional(x) = false
functional(::Union{CPUDevice, Type{<:CPUDevice}}) = true

"""
    loaded(x::AbstractDevice) -> Bool
    loaded(::Type{<:AbstractDevice}) -> Bool

Checks if the trigger package for the device is loaded. Trigger packages are as follows:

  - `CUDA.jl` and `cuDNN.jl` (or just `LuxCUDA.jl`) for NVIDIA CUDA Support.
  - `AMDGPU.jl` for AMD GPU ROCM Support.
  - `Metal.jl` for Apple Metal GPU Support.
  - `oneAPI.jl` for Intel oneAPI GPU Support.
"""
loaded(x) = false
loaded(::Union{CPUDevice, Type{<:CPUDevice}}) = true

# Order is important here
const GPU_DEVICES = (CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice)

const GPU_DEVICE = Ref{Union{Nothing, AbstractDevice}}(nothing)

"""
    reset_gpu_device!()

Resets the selected GPU device. This is useful when automatic GPU selection needs to be
run again.
"""
reset_gpu_device!() = (GPU_DEVICE[] = nothing)

"""
    supported_gpu_backends() -> Tuple{String, ...}

Return a tuple of supported GPU backends.

!!! warning

    This is not the list of functional backends on the system, but rather backends which
    `MLDataDevices.jl` supports.
"""
supported_gpu_backends() = map(Internal.get_device_name, GPU_DEVICES)

"""
    gpu_device(device_id::Union{Nothing, Integer}=nothing;
        force_gpu_usage::Bool=false) -> AbstractDevice()

Selects GPU device based on the following criteria:

 1. If `gpu_backend` preference is set and the backend is functional on the system, then
    that device is selected.
 2. Otherwise, an automatic selection algorithm is used. We go over possible device
    backends in the order specified by `supported_gpu_backends()` and select the first
    functional backend.
 3. If no GPU device is functional and  `force_gpu_usage` is `false`, then `cpu_device()` is
    invoked.
 4. If nothing works, an error is thrown.

## Arguments

  - `device_id::Union{Nothing, Integer}`: The device id to select. If `nothing`, then we return
    the last selected device or if none was selected then we run the autoselection and
    choose the current device using `CUDA.device()` or `AMDGPU.device()` or similar. If
    `Integer`, then we select the device with the given id. Note that this is `1`-indexed, in
    contrast to the `0`-indexed `CUDA.jl`. For example, `id = 4` corresponds to
    `CUDA.device!(3)`.

!!! warning

    `device_id` is only applicable for `CUDA` and `AMDGPU` backends. For `Metal`, `oneAPI`
    and `CPU` backends, `device_id` is ignored and a warning is printed.

!!! warning

    `gpu_device` won't select a CUDA device unless both CUDA.jl and cuDNN.jl are loaded.
    This is to ensure that deep learning operations work correctly.
    Nonetheless, if cuDNN is not loaded you can still manually create a
    `CUDADevice` object and use it (e.g. `dev = CUDADevice()`).

## Keyword Arguments

  - `force_gpu_usage::Bool`: If `true`, then an error is thrown if no functional GPU
    device is found.
"""
function gpu_device(device_id::Union{Nothing, <:Integer}=nothing;
        force_gpu_usage::Bool=false)::AbstractDevice
    device_id == 0 && throw(ArgumentError("`device_id` is 1-indexed."))

    if GPU_DEVICE[] !== nothing
        dev = GPU_DEVICE[]
        if device_id === nothing
            force_gpu_usage &&
                !(dev isa AbstractGPUDevice) &&
                throw(Internal.DeviceSelectionException())
            return dev
        else
            selected_device_id = Internal.get_device_id(dev)
            selected_device_id !== nothing && selected_device_id == device_id && return dev
        end
    end

    device_type = Internal.get_gpu_device(; force_gpu_usage)
    device = Internal.with_device(device_type, device_id)
    GPU_DEVICE[] = device

    return device
end

"""
    gpu_backend!() = gpu_backend!("")
    gpu_backend!(backend) = gpu_backend!(string(backend))
    gpu_backend!(backend::AbstractGPUDevice)
    gpu_backend!(backend::String)

Creates a `LocalPreferences.toml` file with the desired GPU backend.

If `backend == ""`, then the `gpu_backend` preference is deleted. Otherwise, `backend` is
validated to be one of the possible backends and the preference is set to `backend`.

If a new backend is successfully set, then the Julia session must be restarted for the
change to take effect.
"""
gpu_backend!(backend) = gpu_backend!(string(backend))
gpu_backend!(backend::AbstractGPUDevice) = gpu_backend!(Internal.get_device_name(backend))
gpu_backend!() = gpu_backend!("")
function gpu_backend!(backend::String)
    if backend == ""
        @delete_preferences!("gpu_backend")
        @info "Deleted the local preference for `gpu_backend`. Restart Julia to use the \
               new backend."
        return
    end

    allowed_backends = supported_gpu_backends()

    set_backend = @load_preference("gpu_backend", nothing)
    if set_backend == backend
        @info "GPU backend is already set to $backend. No action is required."
        return
    end

    if backend ∉ allowed_backends
        throw(ArgumentError("Invalid backend: $backend. Valid backends are $allowed_backends."))
    end

    @set_preferences!("gpu_backend"=>backend)
    @info "GPU backend has been set to $backend. Restart Julia to use the new backend."
    return
end

"""
    cpu_device() -> CPUDevice()

Return a `CPUDevice` object which can be used to transfer data to CPU.
"""
cpu_device() = CPUDevice()

"""
    xla_device() -> XLADevice()

Return a `XLADevice` object.

!!! danger

    This is an experimental feature and might change without deprecations
"""
function xla_device()
    @assert loaded(XLADevice)&&functional(XLADevice) "`XLADevice` is not loaded or not \
                                                      functional. Load `Reactant.jl` \
                                                      before calling this function."
    return XLADevice()
end

"""
    default_device_rng(::AbstractDevice)

Returns the default RNG for the device. This can be used to directly generate parameters
and states on the device using
[WeightInitializers.jl](https://github.com/LuxDL/WeightInitializers.jl).
"""
function default_device_rng(D::AbstractDevice)
    return error("""`default_device_rng` not implemented for `$(typeof(D))`. This is \
           either because:

           1. The default RNG for this device is not known / officially provided.
           2. The trigger package for the device ($(Internal.get_device_name(D)).jl) is \
              not loaded.
           """)
end
default_device_rng(::CPUDevice) = Random.default_rng()

const GET_DEVICE_ADMONITIONS = """
!!! note

    Trigger Packages must be loaded for this to return the correct device.

!!! warning

    RNG types currently don't participate in device determination. We will remove this
    restriction in the future.
"""

# Query Device from Array
"""
    get_device(x) -> dev::AbstractDevice | Exception | Nothing

If all arrays (on the leaves of the structure) are on the same device, we return that
device. Otherwise, we throw an error. If the object is device agnostic, we return `nothing`.

$(GET_DEVICE_ADMONITIONS)

See also [`get_device_type`](@ref) for a faster alternative that can be used for dispatch
based on device type.
"""
function get_device end

"""
    get_device_type(x) -> Type{<:AbstractDevice} | Exception | Type{Nothing}

Similar to [`get_device`](@ref) but returns the type of the device instead of the device
itself. This value is often a compile time constant and is recommended to be used instead
of [`get_device`](@ref) where ever defining dispatches based on the device type.

$(GET_DEVICE_ADMONITIONS)
"""
function get_device_type end

# Set the device
const SET_DEVICE_DOCS = """
Set the device for the given type. This is a no-op for `CPUDevice`. For `CUDADevice`
and `AMDGPUDevice`, it prints a warning if the corresponding trigger package is not
loaded.
    
Currently, `MetalDevice` and `oneAPIDevice` don't support setting the device.
"""

const SET_DEVICE_DANGER = """
!!! danger

    This specific function should be considered experimental at this point and is currently
    provided to support distributed training in Lux. As such please use
    `Lux.DistributedUtils` instead of using this function.
"""

"""
    set_device!(T::Type{<:AbstractDevice}, dev_or_id)

$SET_DEVICE_DOCS

## Arguments

  - `T::Type{<:AbstractDevice}`: The device type to set.
  - `dev_or_id`: Can be the device from the corresponding package. For example for CUDA it
    can be a `CuDevice`. If it is an integer, it is the device id to set. This is
    `1`-indexed.

$SET_DEVICE_DANGER
"""
function set_device!(::Type{T}, dev_or_id) where {T <: AbstractDevice}
    T === CUDADevice && @warn "`CUDA.jl` hasn't been loaded. Ignoring the device setting."
    T === AMDGPUDevice &&
        @warn "`AMDGPU.jl` hasn't been loaded. Ignoring the device setting."
    T === MetalDevice &&
        @warn "Support for Multi Device Metal hasn't been implemented yet. Ignoring the device setting."
    T === oneAPIDevice &&
        @warn "Support for Multi Device oneAPI hasn't been implemented yet. Ignoring the device setting."
    T === CPUDevice &&
        @warn "Setting device for `CPUDevice` doesn't make sense. Ignoring the device setting."
    T === XLADevice &&
        @warn "Setting device for `XLADevice` hasn't been implemented yet. Ignoring the device setting."
    return
end

"""
    set_device!(T::Type{<:AbstractDevice}, ::Nothing, rank::Integer)

$SET_DEVICE_DOCS

## Arguments

  - `T::Type{<:AbstractDevice}`: The device type to set.
  - `rank::Integer`: Local Rank of the process. This is applicable for distributed training and
    must be `0`-indexed.

$SET_DEVICE_DANGER
"""
function set_device!(::Type{T}, ::Nothing, rank::Integer) where {T <: AbstractDevice}
    return set_device!(T, rank)
end

# Dispatches for Different Data Structures
# Abstract Array / Tuples / NamedTuples have special fast paths to facilitate type stability
# For all other types we rely on fmap which means we lose type stability.
# For Lux, typically models only has these 3 datastructures so we should be mostly fine.
for (dev) in (:CPU, :CUDA, :AMDGPU, :Metal, :oneAPI, :XLA)
    ldev = Symbol(dev, :Device)
    @eval begin
        function (D::$(ldev))(x::AbstractArray{T}) where {T}
            return (isbitstype(T) || Internal.special_aos(x)) ? Adapt.adapt(D, x) :
                   map(D, x)
        end
        (D::$(ldev))(x::Union{Tuple, NamedTuple}) = map(D, x)
        function (D::$(ldev))(x)
            Functors.isleaf(x) && return Adapt.adapt(D, x)
            return Functors.fmap(D, x)
        end
    end
end

for op in (:get_device, :get_device_type)
    @eval function $(op)(x)
        hasmethod(Internal.$(op), Tuple{typeof(x)}) && return Internal.$(op)(x)
        return mapreduce(Internal.$(op), Internal.combine_devices, fleaves(x))
    end
end

# Adapt Interface
Adapt.adapt_storage(::CPUDevice, x::AbstractArray) = Adapt.adapt(Array, x)
Adapt.adapt_storage(::CPUDevice, rng::AbstractRNG) = rng

for T in (AMDGPUDevice, CUDADevice, MetalDevice, oneAPIDevice, XLADevice)
    @eval begin
        function Adapt.adapt_storage(to::$(T), ::Random.TaskLocalRNG)
            return default_device_rng(to)
        end
        Adapt.adapt_storage(::$(T), rng::AbstractRNG) = rng
    end
end

Adapt.adapt_storage(::CPUDevice, x::AbstractRange) = x
Adapt.adapt_storage(::XLADevice, x::AbstractRange) = x
# Prevent Ambiguity
for T in (AMDGPUDevice, AMDGPUDevice{Nothing}, CUDADevice,
    CUDADevice{Nothing}, MetalDevice, oneAPIDevice)
    @eval Adapt.adapt_storage(to::$(T), x::AbstractRange) = Adapt.adapt(to, collect(x))
end

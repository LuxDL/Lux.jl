const EltypeAdaptorType = Union{Missing,Nothing,<:AbstractFloat}

struct CPUDevice{T<:EltypeAdaptorType} <: AbstractCPUDevice end
CPUDevice() = CPUDevice{Missing}()

struct CUDADevice{D,T<:EltypeAdaptorType} <: AbstractGPUDevice
    device::D
end
CUDADevice() = CUDADevice{Nothing,Missing}(nothing)
CUDADevice(device) = CUDADevice{typeof(device),Missing}(device)

struct AMDGPUDevice{D,T<:EltypeAdaptorType} <: AbstractGPUDevice
    device::D
end
AMDGPUDevice() = AMDGPUDevice{Nothing,Missing}(nothing)
AMDGPUDevice(device) = AMDGPUDevice{typeof(device),Missing}(device)

struct MetalDevice{T<:EltypeAdaptorType} <: AbstractGPUDevice end
MetalDevice() = MetalDevice{Missing}()

struct oneAPIDevice{T<:EltypeAdaptorType} <: AbstractGPUDevice end
oneAPIDevice() = oneAPIDevice{Missing}()

struct OpenCLDevice{T<:EltypeAdaptorType} <: AbstractGPUDevice end
OpenCLDevice() = OpenCLDevice{Missing}()

struct ReactantDevice{C,D,S,T<:EltypeAdaptorType,TN} <: AbstractAcceleratorDevice
    client::C
    device::D
    sharding::S
end
function ReactantDevice()
    return ReactantDevice{Missing,Missing,Missing,Missing,Union{}}(
        missing, missing, missing
    )
end
function ReactantDevice(client, device, sharding, _::Type{TN}=Union{}) where {TN}
    return ReactantDevice{typeof(client),typeof(device),typeof(sharding),Missing,TN}(
        client, device, sharding
    )
end

function with_track_numbers(
    dev::ReactantDevice{C,D,S,T,Union{}}, _::Type{TN}
) where {C,D,S,T,TN}
    return ReactantDevice{C,D,S,T,TN}(dev.client, dev.device, dev.sharding)
end

# Helper functions to get the eltype from device types
Base.eltype(::CPUDevice{T}) where {T} = T
Base.eltype(::CUDADevice{D,T}) where {D,T} = T
Base.eltype(::AMDGPUDevice{D,T}) where {D,T} = T
Base.eltype(::MetalDevice{T}) where {T} = T
Base.eltype(::oneAPIDevice{T}) where {T} = T
Base.eltype(::OpenCLDevice{T}) where {T} = T
Base.eltype(::ReactantDevice{C,D,S,T}) where {C,D,S,T} = T

# Helper functions to create devices with specific eltypes
with_eltype(::CPUDevice, ::Nothing) = CPUDevice{Nothing}()
with_eltype(::CPUDevice, ::Missing) = CPUDevice{Missing}()
with_eltype(::CPUDevice, ::Type{T}) where {T<:AbstractFloat} = CPUDevice{T}()

with_eltype(dev::CUDADevice{D}, ::Nothing) where {D} = CUDADevice{D,Nothing}(dev.device)
with_eltype(dev::CUDADevice{D}, ::Missing) where {D} = CUDADevice{D,Missing}(dev.device)
function with_eltype(dev::CUDADevice{D}, ::Type{T}) where {D,T<:AbstractFloat}
    return CUDADevice{D,T}(dev.device)
end

with_eltype(dev::AMDGPUDevice{D}, ::Nothing) where {D} = AMDGPUDevice{D,Nothing}(dev.device)
with_eltype(dev::AMDGPUDevice{D}, ::Missing) where {D} = AMDGPUDevice{D,Missing}(dev.device)
function with_eltype(dev::AMDGPUDevice{D}, ::Type{T}) where {D,T<:AbstractFloat}
    return AMDGPUDevice{D,T}(dev.device)
end

with_eltype(::MetalDevice, ::Nothing) = MetalDevice{Nothing}()
with_eltype(::MetalDevice, ::Missing) = MetalDevice{Missing}()
with_eltype(::MetalDevice, ::Type{T}) where {T<:AbstractFloat} = MetalDevice{T}()

with_eltype(::oneAPIDevice, ::Nothing) = oneAPIDevice{Nothing}()
with_eltype(::oneAPIDevice, ::Missing) = oneAPIDevice{Missing}()
function with_eltype(::oneAPIDevice, ::Type{T}) where {T<:AbstractFloat}
    return oneAPIDevice{T}()
end

with_eltype(::OpenCLDevice, ::Nothing) = OpenCLDevice{Nothing}()
with_eltype(::OpenCLDevice, ::Missing) = OpenCLDevice{Missing}()
function with_eltype(::OpenCLDevice, ::Type{T}) where {T<:AbstractFloat}
    return OpenCLDevice{T}()
end

function with_eltype(dev::ReactantDevice{C,D,S,<:Any,TN}, ::Missing) where {C,D,S,TN}
    return ReactantDevice{C,D,S,Missing,TN}(dev.client, dev.device, dev.sharding)
end
function with_eltype(dev::ReactantDevice{C,D,S,<:Any,TN}, ::Nothing) where {C,D,S,TN}
    return ReactantDevice{C,D,S,Nothing,TN}(dev.client, dev.device, dev.sharding)
end
function with_eltype(
    dev::ReactantDevice{C,D,S,<:Any,TN}, ::Type{T}
) where {C,D,S,TN,T<:AbstractFloat}
    return ReactantDevice{C,D,S,T,TN}(dev.client, dev.device, dev.sharding)
end

function Base.:(==)(
    x::ReactantDevice{<:Any,<:Any,<:Any,T1,TN1}, y::ReactantDevice{<:Any,<:Any,<:Any,T2,TN2}
) where {T1,T2,TN1,TN2}
    if x.client !== missing && y.client !== missing && x.client.client != y.client.client
        return false
    end

    if (
        x.device !== missing &&
        x.device !== nothing && # can be nothing if objects are sharded
        y.device !== missing &&
        y.device !== nothing && # can be nothing if objects are sharded
        x.device.device != y.device.device
    )
        return false
    end

    T1 === Missing && return T2 === Missing || T2 === Nothing
    T2 === Missing && return T1 === Missing || T1 === Nothing
    return T1 === T2 && TN1 === TN2
end

# XXX: Deprecate in v2
const XLADevice = ReactantDevice

# Fallback for when we don't know the device type
struct UnknownDevice <: AbstractDevice end

"""
    functional(x::AbstractDevice) -> Bool
    functional(::Type{<:AbstractDevice}) -> Bool

Checks if the device is functional. This is used to determine if the device can be used for
computation. Note that even if the backend is loaded (as checked via
[`MLDataDevices.loaded`](@ref)), the device may not be functional.

Note that while this function is not exported, it is considered part of the public API.
"""
functional(x) = false
functional(::Union{CPUDevice,Type{<:CPUDevice}}) = true

"""
    loaded(x::AbstractDevice) -> Bool
    loaded(::Type{<:AbstractDevice}) -> Bool

Checks if the trigger package for the device is loaded. Trigger packages are as follows:

  - `CUDA.jl` and `cuDNN.jl` (or just `LuxCUDA.jl`) for NVIDIA CUDA Support.
  - `AMDGPU.jl` for AMD GPU ROCM Support.
  - `Metal.jl` for Apple Metal GPU Support.
  - `oneAPI.jl` for Intel oneAPI GPU Support.
  - `OpenCL.jl` for OpenCL support.
"""
loaded(x) = false
loaded(::Union{CPUDevice,Type{<:CPUDevice}}) = true

# Order is important here
const GPU_DEVICES = (CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice, OpenCLDevice)

const GPU_DEVICE = Ref{Union{Nothing,AbstractDevice}}(nothing)

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
    gpu_device(
        eltype::Union{Missing, Nothing, Type{<:AbstractFloat}}=missing;
        kwargs...
    ) -> AbstractDevice
    gpu_device(
        device_id::Union{Nothing, Integer}=nothing
        eltype::Union{Missing, Nothing, Type{<:AbstractFloat}}=missing;
        force::Bool=false
    ) -> AbstractDevice

Selects GPU device based on the following criteria:

 1. If `gpu_backend` preference is set and the backend is functional on the system, then
    that device is selected.
 2. Otherwise, an automatic selection algorithm is used. We go over possible device
    backends in the order specified by `supported_gpu_backends()` and select the first
    functional backend.
 3. If no GPU device is functional and  `force` is `false`, then `cpu_device()` is
    invoked.
 4. If nothing works, an error is thrown.

## Arguments

  - `device_id::Union{Nothing, Integer}`: The device id to select. If `nothing`, then
    we return the last selected device or if none was selected then we run the autoselection
    and choose the current device using `CUDA.device()` or `AMDGPU.device()` or similar. If
    `Integer`, then we select the device with the given id. Note that this is `1`-indexed,
    in contrast to the `0`-indexed `CUDA.jl`. For example, `id = 4` corresponds to
    `CUDA.device!(3)`.
  - `eltype::Union{Missing, Nothing, Type{<:AbstractFloat}}`: The element type to use for
    the device.
    - `missing` (default): Device specific. For `CUDADevice` this calls `CUDA.cu(x)`,
      for `AMDGPUDevice` this calls `AMDGPU.roc(x)`, for `MetalDevice` this calls
      `Metal.mtl(x)`, for `oneAPIDevice` this calls `oneArray(x)`, for `OpenCLDevice` this calls `CLArray(x)`.
    - `nothing`: Preserves the original element type.
    - `Type{<:AbstractFloat}`: Converts floating-point arrays to the specified type.

!!! warning

    `device_id` is only applicable for `CUDA` and `AMDGPU` backends. For `Metal`, `oneAPI`, `OpenCL`
    and `CPU` backends, `device_id` is ignored and a warning is printed.

!!! warning

    `gpu_device` won't select a CUDA device unless both CUDA.jl and cuDNN.jl are loaded.
    This is to ensure that deep learning operations work correctly.
    Nonetheless, if cuDNN is not loaded you can still manually create a
    `CUDADevice` object and use it (e.g. `dev = CUDADevice()`).

## Keyword Arguments

  - `force::Bool`: If `true`, then an error is thrown if no functional GPU
    device is found.
"""
function gpu_device(eltype::EltypeAdaptorType; kwargs...) where {EltypeAdaptorType}
    return gpu_device(nothing, eltype; kwargs...)
end

function gpu_device(
    device_id::Union{Nothing,<:Integer}=nothing,
    eltype::T=missing;
    force::Bool=false,
    force_gpu_usage::Union{Missing,Bool}=missing,
)::AbstractDevice where {T}
    if force_gpu_usage !== missing
        Base.depwarn(
            "`force_gpu_usage` is deprecated and will be removed in v2. Use \
             `force` instead.",
            :gpu_device,
        )
        force = force_gpu_usage
    end

    device_id == 0 && throw(ArgumentError("`device_id` is 1-indexed."))

    if GPU_DEVICE[] !== nothing
        dev = GPU_DEVICE[]
        if device_id === nothing
            if force && !(dev isa AbstractGPUDevice)
                throw(Internal.DeviceSelectionException("GPU"))
            end
            return with_eltype(dev, eltype)
        else
            selected_device_id = Internal.get_device_id(dev)
            if selected_device_id !== nothing && selected_device_id == device_id
                return with_eltype(dev, eltype)
            end
        end
    end

    device_type = Internal.get_gpu_device(; force)
    device = Internal.with_device(device_type, device_id)
    GPU_DEVICE[] = device
    return with_eltype(device, eltype)
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
        return nothing
    end

    allowed_backends = supported_gpu_backends()

    set_backend = @load_preference("gpu_backend", nothing)
    if set_backend == backend
        @info "GPU backend is already set to $backend. No action is required."
        return nothing
    end

    if backend ∉ allowed_backends
        throw(
            ArgumentError(
                "Invalid backend: $backend. Valid backends are $allowed_backends."
            ),
        )
    end

    @set_preferences!("gpu_backend" => backend)
    @info "GPU backend has been set to $backend. Restart Julia to use the new backend."
    return nothing
end

"""
    cpu_device(eltype=missing) -> CPUDevice

Return a `CPUDevice` object which can be used to transfer data to CPU.

The `eltype` parameter controls element type conversion:

  - `missing/nothing` (default): Preserves the original element type
  - `Type{<:AbstractFloat}`: Converts floating-point arrays to the specified type
"""
cpu_device(eltype::T=missing) where {T} = with_eltype(CPUDevice(), eltype)

"""
    reactant_device(;
        force::Bool=false, client=missing, device=missing, sharding=missing, eltype=missing,
        track_numbers::Type{TN}=Union{}
    ) -> Union{ReactantDevice, CPUDevice}

Return a `ReactantDevice` object if functional. Otherwise, throw an error if `force` is
`true`. Falls back to `CPUDevice` if `force` is `false`.

`client` and `device` are used to specify the client and particular device to use. If not
specified, then the default client and index are used.

`sharding` is used to specify the sharding strategy. If a
`Reactant.Sharding.AbstractSharding` is specified, then we use it to shard all abstract
arrays. Alternatively, pass in a `IdDict` to specify the sharding for specific leaves.

`track_numbers` can be specified to convert numbers of specified subtypes to be traced.

The `eltype` parameter controls element type conversion:

  - `missing/nothing` (default): Preserves the original element type
  - `Type{<:AbstractFloat}`: Converts floating-point arrays to the specified type
"""
function reactant_device(
    eltype::T=missing;
    force::Bool=false,
    client=missing,
    device=missing,
    sharding=missing,
    track_numbers::Type{TN}=Union{},
) where {T,TN}
    msg = "`ReactantDevice` is not loaded or not functional. Load `Reactant.jl` before \
           calling this function. Defaulting to CPU."
    if loaded(ReactantDevice)
        if functional(ReactantDevice)
            return with_track_numbers(
                with_eltype(ReactantDevice(client, device, sharding), eltype), track_numbers
            )
        end
        msg = "`ReactantDevice` is loaded but not functional. Defaulting to CPU."
    end
    force && throw(Internal.DeviceSelectionException("Reactant"))
    @warn msg maxlog = 1
    return cpu_device(eltype)
end

Base.@deprecate xla_device(; kwargs...) reactant_device(; kwargs...)

"""
    default_device_rng(::AbstractDevice)

Returns the default RNG for the device. This can be used to directly generate parameters
and states on the device using
[WeightInitializers.jl](https://github.com/LuxDL/WeightInitializers.jl).
"""
function default_device_rng(D::AbstractDevice)
    return error(
        """`default_device_rng` not implemented for `$(typeof(D))`. This is \
        either because:

        1. The default RNG for this device is not known / officially provided.
        2. The trigger package for the device ($(Internal.get_device_name(D)).jl) is \
        not loaded.
        """
    )
end
default_device_rng(::CPUDevice) = Random.default_rng()

const GET_DEVICE_ADMONITIONS = """
!!! note

    Trigger Packages must be loaded for this to return the correct device.
"""

# Query Device from Array
"""
    get_device(x) -> dev::AbstractDevice | Exception | Nothing

If all arrays (on the leaves of the structure) are on the same device, we return that
device. Otherwise, we throw an error. If the object is device agnostic, we return `nothing`.

$(GET_DEVICE_ADMONITIONS)

## Special Retuened Values

  - `nothing` -- denotes that the object is device agnostic. For example, scalar, abstract
    range, etc.
  - `UnknownDevice()` -- denotes that the device type is unknown.

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

## Special Retuened Values

  - `Nothing` -- denotes that the object is device agnostic. For example, scalar, abstract
    range, etc.
  - `UnknownDevice` -- denotes that the device type is unknown.
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
function set_device!(::Type{T}, dev_or_id) where {T<:AbstractDevice}
    T === CUDADevice && @warn "`CUDA.jl` hasn't been loaded. Ignoring the device setting."
    T === AMDGPUDevice &&
        @warn "`AMDGPU.jl` hasn't been loaded. Ignoring the device setting."
    T === MetalDevice &&
        @warn "Support for Multi Device Metal hasn't been implemented yet. Ignoring the device setting."
    T === oneAPIDevice &&
        @warn "Support for Multi Device oneAPI hasn't been implemented yet. Ignoring the device setting."
    T === OpenCLDevice &&
        @warn "Support for Multi Device OpenCL hasn't been implemented yet. Ignoring the device setting."
    T === CPUDevice &&
        @warn "Setting device for `CPUDevice` doesn't make sense. Ignoring the device setting."
    T === ReactantDevice &&
        @warn "Setting device for `ReactantDevice` hasn't been implemented yet. Ignoring the device setting."
    return nothing
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
function set_device!(::Type{T}, ::Nothing, rank::Integer) where {T<:AbstractDevice}
    return set_device!(T, rank)
end

# Dispatches for Different Data Structures
(D::AbstractDevice)(x) = Functors.fmap(Base.Fix1(Adapt.adapt, D), x; exclude=isleaf)

for op in (:get_device, :get_device_type)
    @eval function $(op)(x)
        Internal.fast_structure(x) && return Internal.$(op)(x)
        return mapreduce(
            Internal.$(op),
            Internal.combine_devices,
            fleaves(x; exclude=isleaf);
            init=$(op == :get_device ? nothing : Nothing),
        )
    end
end

# Adapt Interface
function Adapt.adapt_storage(dev::CPUDevice{Missing}, x::AbstractArray)
    get_device_type(x) <: CPUDevice && return x
    return Array(x)
end

function Adapt.adapt_storage(dev::CPUDevice{Nothing}, x::AbstractArray)
    get_device_type(x) <: CPUDevice && return x
    return Array(x)  # Preserve eltype
end

function Adapt.adapt_storage(dev::CPUDevice{T}, x::AbstractArray) where {T<:AbstractFloat}
    get_device_type(x) <: CPUDevice && eltype(x) == T && return x
    x_cpu = Array(x)

    # Only convert floating-point and complex floating-point types
    ET = eltype(x_cpu)
    if ET <: AbstractFloat
        return Array{T}(x_cpu)
    elseif ET <: Complex{<:AbstractFloat}
        return Array{Complex{T}}(x_cpu)
    else
        return x_cpu  # Don't convert non-floating point types
    end
end

Adapt.adapt_storage(to::AbstractDevice, ::Random.TaskLocalRNG) = default_device_rng(to)
Adapt.adapt_storage(::AbstractDevice, rng::AbstractRNG) = rng

"""
    isleaf(x) -> Bool

Returns `true` if `x` is a leaf node in the data structure.

Defining `MLDataDevices.isleaf(x::T) = true` for custom types
can be used to customize the behavior the data movement behavior
when an object with nested structure containing the type is transferred to a device.

`Adapt.adapt_structure(::AbstractDevice, x::T)` or
`Adapt.adapt_structure(::AbstractDevice, x::T)` will be called during
data movement if `isleaf(x::T)`.

If `MLDataDevices.isleaf(x::T)` is not defined, then it will fall back to
`Functors.isleaf(x)`.
"""
isleaf(x) = Functors.isleaf(x)

function isleaf(x::AbstractArray{T}) where {T}
    parent(x) !== x && return Functors.isleaf(x)
    return isbitstype(T) || T <: Number # BigFloat and such are not bitstype
end

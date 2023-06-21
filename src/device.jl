import Base: PkgId, UUID

abstract type AbstractLuxDevice end
abstract type AbstractLuxGPUDevice <: AbstractLuxDevice end

struct LuxCPUDevice <: AbstractLuxDevice end

Base.@kwdef struct LuxCUDADevice <: AbstractLuxGPUDevice
    name::String = "CUDA"
    pkgid::PkgId = PkgId(UUID("d0bbae9a-e099-4d5b-a835-1c6931763bda"), "LuxCUDA")
end

Base.@kwdef struct LuxAMDGPUDevice <: AbstractLuxGPUDevice
    name::String = "AMDGPU"
    pkgid::PkgId = PkgId(UUID("83120cb1-ca15-4f04-bf3b-6967d2e6b60b"), "LuxAMDGPU")
end

struct LuxDeviceSelectionException <: Exception end

function Base.showerror(io::IO, e::LuxDeviceSelectionException)
    print(io, "LuxDeviceSelectionException(No functional GPU device found!!)")
    if !TruncatedStacktraces.VERBOSE[]
        println(io, TruncatedStacktraces.VERBOSE_MSG)
    end
end

@generated function _get_device_name(t::T) where {T <: AbstractLuxDevice}
    return hasfield(T, :name) ? :(t.name) : :("")
end

@generated function _get_trigger_pkgid(t::T) where {T <: AbstractLuxDevice}
    return hasfield(T, :pkgid) ? :(t.pkgid) :
           :(PkgId(UUID("b2108857-7c20-44ae-9111-449ecde12c47"), "Lux"))
end

const GPU_DEVICES = (LuxCUDADevice(), LuxAMDGPUDevice())  # Order is important here

const GPU_DEVICE = Ref{Union{Nothing, AbstractLuxDevice}}(nothing)

"""
    supported_gpu_backends() -> Tuple{String, ...}

Return a tuple of supported GPU backends.

!!! warning

    This is not the list of functional backends on the system, but rather backends which
    `Lux.jl` supports.
"""
supported_gpu_backends() = map(_get_device_name, GPU_DEVICES)

"""
    get_gpu_device(; force_gpu_usage::Bool=false) -> AbstractLuxDevice()

Selects GPU device based on the following criteria:

    1. If `gpu_backend` preference is set and the backend is functional on the system, then
       that device is selected.
    2. Otherwise, an automatic selection algorithm is used. We go over possible device
       backends in the order specified by `supported_gpu_backends()` and select the first
       functional backend.
    3. If no GPU device is functional and  force_gpu_usage` is `false`, then
       `get_cpu_device()` is invoked.
    4. If nothing works, an error is thrown.
"""
function get_gpu_device(; force_gpu_usage::Bool=false)::AbstractLuxDevice
    if !ACCELERATOR_STATE_CHANGED[]
        if GPU_DEVICE[] !== nothing
            force_gpu_usage &&
                !(GPU_DEVICE[] isa AbstractLuxGPUDevice) &&
                throw(LuxDeviceSelectionException())
            return GPU_DEVICE[]
        end
    end

    device = _get_gpu_device_impl(; force_gpu_usage)
    ACCELERATOR_STATE_CHANGED[] = false
    GPU_DEVICE[] = device

    return device
end

function _get_gpu_device_impl(; force_gpu_usage::Bool)
    backend = @load_preference("gpu_backend", nothing)

    # If backend set with preferences, use it
    if backend !== nothing
        allowed_backends = supported_gpu_backends()
        idx = findfirst(isequal(backend), allowed_backends)
        if backend ∉ allowed_backends
            @warn """
            `gpu_backend` preference is set to $backend, which is not a valid backend.
            Valid backends are $allowed_backends.
            Defaulting to automatic GPU Backend selection.
            """ maxlog=1
        else
            @debug "Using GPU backend set in preferences: $backend."
            return GPU_DEVICES[idx]
        end
    end

    @debug "Running automatic GPU backend selection..."
    for device in GPU_DEVICES
        if haskey(Base.loaded_modules, device.pkgid)
            @debug "Trying backend: $(_get_device_name(device))."
            if getproperty(Base.loaded_modules[device.pkgid], :functional)()
                @debug "Using GPU backend: $(_get_device_name(device))."
                return device
            end
            @debug "GPU backend: $(_get_device_name(device)) is not functional."
        else
            @debug "Trigger package for backend ($(_get_device_name(device))): $((device.pkgid)) not loaded."
        end
    end

    force_gpu_usage ? throw(LuxDeviceSelectionException()) : return get_cpu_device()
end

"""
    get_cpu_device() -> LuxCPUDevice()

Return a `LuxCPUDevice` object which can be used to transfer data to CPU.
"""
@inline get_cpu_device() = LuxCPUDevice()

(::LuxCPUDevice)(x) = fmap(x -> adapt(LuxCPUAdaptor(), x), x; exclude=_isleaf)
(::LuxCUDADevice)(x) = fmap(x -> adapt(LuxCUDAAdaptor(), x), x; exclude=_isleaf)
(::LuxAMDGPUDevice)(x) = fmap(x -> adapt(LuxAMDGPUAdaptor(), x), x; exclude=_isleaf)

function (::AbstractLuxDevice)(::AbstractExplicitLayer)
    throw(ArgumentError("Lux layers are stateless and hence don't participate in device transfers. Apply this function on the parameters and states generated using `Lux.setup`."))
end

# Adapt Interface
abstract type LuxDeviceAdaptor end

struct LuxCPUAdaptor <: LuxDeviceAdaptor end
struct LuxCUDAAdaptor <: LuxDeviceAdaptor end
struct LuxAMDGPUAdaptor <: LuxDeviceAdaptor end

function adapt_storage(::LuxCPUAdaptor,
    x::Union{AbstractRange, SparseArrays.AbstractSparseArray})
    return x
end
adapt_storage(::LuxCPUAdaptor, x::AbstractArray) = adapt(Array, x)
adapt_storage(::LuxCPUAdaptor, rng::AbstractRNG) = rng

_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where {T} = isbitstype(T)
_isbitsarray(x) = false

_isleaf(::AbstractRNG) = true
_isleaf(x) = _isbitsarray(x) || Functors.isleaf(x)

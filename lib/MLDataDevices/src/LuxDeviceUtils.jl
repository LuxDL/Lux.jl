module LuxDeviceUtils

using Functors, LuxCore, Preferences, Random, SparseArrays
import Adapt: adapt, adapt_storage
import Base: PkgId, UUID

using PackageExtensionCompat
function __init__()
    @require_extensions
end

export gpu_backend!, supported_gpu_backends
export gpu_device, cpu_device, LuxCPUDevice, LuxCUDADevice, LuxAMDGPUDevice, LuxMetalDevice
export LuxCPUAdaptor, LuxCUDAAdaptor, LuxAMDGPUAdaptor, LuxMetalAdaptor

const ACCELERATOR_STATE_CHANGED = Ref{Bool}(false)

abstract type AbstractLuxDevice <: Function end
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

Base.@kwdef struct LuxMetalDevice <: AbstractLuxGPUDevice
    name::String = "Metal"
    pkgid::PkgId = PkgId(UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal")
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

# Order is important here
const GPU_DEVICES = (LuxCUDADevice(), LuxAMDGPUDevice(), LuxMetalDevice())

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
    gpu_device(; force_gpu_usage::Bool=false) -> AbstractLuxDevice()

Selects GPU device based on the following criteria:

 1. If `gpu_backend` preference is set and the backend is functional on the system, then
    that device is selected.
 2. Otherwise, an automatic selection algorithm is used. We go over possible device
    backends in the order specified by `supported_gpu_backends()` and select the first
    functional backend.
 3. If no GPU device is functional and  `force_gpu_usage` is `false`, then `cpu_device()` is
    invoked.
 4. If nothing works, an error is thrown.
"""
function gpu_device(; force_gpu_usage::Bool=false)::AbstractLuxDevice
    if !ACCELERATOR_STATE_CHANGED[]
        if GPU_DEVICE[] !== nothing
            force_gpu_usage &&
                !(GPU_DEVICE[] isa AbstractLuxGPUDevice) &&
                throw(LuxDeviceSelectionException())
            return GPU_DEVICE[]
        end
    end

    device = _get_gpu_device(; force_gpu_usage)
    ACCELERATOR_STATE_CHANGED[] = false
    GPU_DEVICE[] = device

    return device
end

function _get_gpu_device(; force_gpu_usage::Bool)
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
            device = GPU_DEVICES[idx]
            if !haskey(Base.loaded_modules, device.pkgid)
                @warn """Trying to use backend: $(_get_device_name(device)) but the trigger package $(device.pkgid) is not loaded.
                    Ignoring the Preferences backend!!!
                    Please load the package and call this function again to respect the Preferences backend.""" maxlog=1
            else
                if getproperty(Base.loaded_modules[dev.pkgid], :functional)()
                    @debug "Using GPU backend: $(_get_device_name(dev))."
                    return dev
                else
                    @warn "GPU backend: $(_get_device_name(device)) set via Preferences.jl is not functional. Defaulting to automatic GPU Backend selection." maxlog=1
                end
            end
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

    if force_gpu_usage
        throw(LuxDeviceSelectionException())
    else
        @warn """No functional GPU backend found! Defaulting to CPU.

                 1. If no GPU is available, nothing needs to be done.
                 2. If GPU is available, load the corresponding trigger package.""" maxlog=1
        return cpu_device()
    end
end

"""
    gpu_backend!() = gpu_backend!("")
    gpu_backend!(backend) = gpu_backend!(string(backend))
    gpu_backend!(backend::AbstractLuxGPUDevice)
    gpu_backend!(backend::String)

Creates a `LocalPreferences.toml` file with the desired GPU backend.

If `backend == ""`, then the `gpu_backend` preference is deleted. Otherwise, `backend` is
validated to be one of the possible backends and the preference is set to `backend`.

If a new backend is successfully set, then the Julia session must be restarted for the
change to take effect.
"""
gpu_backend!(backend) = gpu_backend!(string(backend))
gpu_backend!(backend::AbstractLuxGPUDevice) = gpu_backend!(_get_device_name(backend))
gpu_backend!() = gpu_backend!("")
function gpu_backend!(backend::String)
    if backend == ""
        @delete_preferences!("gpu_backend")
        @info "Deleted the local preference for `gpu_backend`. Restart Julia to use the new backend."
        return
    end

    allowed_backends = supported_gpu_backends()

    set_backend = @load_preference("gpu_backend", nothing)
    if set_backend == backend
        @info "GPU backend is already set to $backend. No action is required."
        return
    end

    @assert backend in allowed_backends "`gpu_backend` must be one of $(allowed_backends)"

    @set_preferences!("gpu_backend"=>backend)
    @info "GPU backend has been set to $backend. Restart Julia to use the new backend."
    return
end

"""
    cpu_device() -> LuxCPUDevice()

Return a `LuxCPUDevice` object which can be used to transfer data to CPU.
"""
@inline cpu_device() = LuxCPUDevice()

(::LuxCPUDevice)(x) = fmap(x -> adapt(LuxCPUAdaptor(), x), x; exclude=_isleaf)
(::LuxCUDADevice)(x) = fmap(x -> adapt(LuxCUDAAdaptor(), x), x; exclude=_isleaf)
(::LuxAMDGPUDevice)(x) = fmap(x -> adapt(LuxAMDGPUAdaptor(), x), x; exclude=_isleaf)
(::LuxMetalDevice)(x) = fmap(x -> adapt(LuxMetalAdaptor(), x), x; exclude=_isleaf)

for dev in (LuxCPUDevice, LuxCUDADevice, LuxAMDGPUDevice, LuxMetalDevice)
    @eval begin
        function (::$dev)(::LuxCore.AbstractExplicitLayer)
            throw(ArgumentError("Lux layers are stateless and hence don't participate in device transfers. Apply this function on the parameters and states generated using `Lux.setup`."))
        end
    end
end

# Adapt Interface
abstract type AbstractLuxDeviceAdaptor end

struct LuxCPUAdaptor <: AbstractLuxDeviceAdaptor end
struct LuxCUDAAdaptor <: AbstractLuxDeviceAdaptor end
struct LuxAMDGPUAdaptor <: AbstractLuxDeviceAdaptor end
struct LuxMetalAdaptor <: AbstractLuxDeviceAdaptor end

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

end

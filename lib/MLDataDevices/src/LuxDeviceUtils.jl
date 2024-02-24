module LuxDeviceUtils

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ChainRulesCore, Functors, LuxCore, Preferences, Random
    import Adapt: adapt, adapt_storage
    import ChainRulesCore as CRC
end

export gpu_backend!, supported_gpu_backends, reset_gpu_device!
export default_device_rng
export gpu_device, cpu_device, LuxCPUDevice, LuxCUDADevice, LuxAMDGPUDevice, LuxMetalDevice
export LuxCPUAdaptor, LuxCUDAAdaptor, LuxAMDGPUAdaptor, LuxMetalAdaptor
export get_device

abstract type AbstractLuxDevice <: Function end
abstract type AbstractLuxGPUDevice <: AbstractLuxDevice end

__is_functional(x) = false
__is_loaded(x) = false

struct LuxCPUDevice <: AbstractLuxDevice end
@kwdef struct LuxCUDADevice{ID} <: AbstractLuxGPUDevice
    device_id::ID = nothing
end
@kwdef struct LuxAMDGPUDevice{ID} <: AbstractLuxGPUDevice
    device_id::ID = nothing
end
struct LuxMetalDevice <: AbstractLuxGPUDevice end

_with_device_id(::Type{LuxCPUDevice}, ::Nothing) = LuxCPUDevice()
function _with_device_id(::Type{LuxCPUDevice}, device_id)
    @warn "`device_id` is not applicable for `LuxCPUDevice`." maxlog=1
    return LuxCPUDevice()
end

_with_device_id(::Type{LuxMetalDevice}, ::Nothing) = LuxMetalDevice()
function _with_device_id(::Type{LuxMetalDevice}, device_id)
    @warn "`device_id` is not applicable for `LuxMetalDevice`." maxlog=1
    return LuxMetalDevice()
end

_get_adaptor(::LuxCPUDevice) = LuxCPUAdaptor()
_get_adaptor(dev::LuxCUDADevice) = LuxCUDAAdaptor(dev.device_id)
_get_adaptor(dev::LuxAMDGPUDevice) = LuxAMDGPUAdaptor(dev.device_id)
_get_adaptor(::LuxMetalDevice) = LuxMetalAdaptor()

__is_functional(::Union{LuxCPUDevice, Type{<:LuxCPUDevice}}) = true
__is_loaded(::Union{LuxCPUDevice, Type{<:LuxCPUDevice}}) = true

_get_device_name(::Union{LuxCPUDevice, Type{<:LuxCPUDevice}}) = "CPU"
_get_device_name(::Union{LuxCUDADevice, Type{<:LuxCUDADevice}}) = "CUDA"
_get_device_name(::Union{LuxAMDGPUDevice, Type{<:LuxAMDGPUDevice}}) = "AMDGPU"
_get_device_name(::Union{LuxMetalDevice, Type{<:LuxMetalDevice}}) = "Metal"

_get_triggerpkg_name(::Union{LuxCPUDevice, Type{<:LuxCPUDevice}}) = ""
_get_triggerpkg_name(::Union{LuxCUDADevice, Type{<:LuxCUDADevice}}) = "LuxCUDA"
_get_triggerpkg_name(::Union{LuxAMDGPUDevice, Type{<:LuxAMDGPUDevice}}) = "LuxAMDGPU"
_get_triggerpkg_name(::Union{LuxMetalDevice, Type{<:LuxMetalDevice}}) = "Metal"

Base.show(io::IO, dev::AbstractLuxDevice) = print(io, nameof(dev))

struct LuxDeviceSelectionException <: Exception end

function Base.showerror(io::IO, ::LuxDeviceSelectionException)
    return print(io, "LuxDeviceSelectionException(No functional GPU device found!!)")
end

# Order is important here
const GPU_DEVICES = (LuxCUDADevice, LuxAMDGPUDevice, LuxMetalDevice)

const GPU_DEVICE = Ref{Union{Nothing, AbstractLuxDevice}}(nothing)

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
    `Lux.jl` supports.

!!! danger

    `Metal.jl` support is **extremely** experimental and most things are not expected to
    work.
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
function gpu_device(device_id=nothing; force_gpu_usage::Bool=false)::AbstractLuxDevice
    if GPU_DEVICE[] !== nothing
        force_gpu_usage && !(GPU_DEVICE[] isa AbstractLuxGPUDevice) &&
            throw(LuxDeviceSelectionException())
        return GPU_DEVICE[]
    end

    device_type = _get_gpu_device(; force_gpu_usage)
    device = _with_device_id(device_type, device_id)
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
            @warn "`gpu_backend` preference is set to $backend, which is not a valid \
                    backend. Valid backends are $allowed_backends. Defaulting to automatic \
                    GPU Backend selection." maxlog=1
        else
            @debug "Using GPU backend set in preferences: $backend."
            device = GPU_DEVICES[idx]
            if !__is_loaded(device)
                @warn "Trying to use backend: $(_get_device_name(device)) but the trigger \
                       package $(device.pkgid) is not loaded. Ignoring the Preferences \
                       backend!!! Please load the package and call this function again to \
                       respect the Preferences backend." maxlog=1
            else
                if __is_functional(device)
                    @debug "Using GPU backend: $(_get_device_name(device))."
                    return device
                else
                    @warn "GPU backend: $(_get_device_name(device)) set via Preferences.jl \
                           is not functional. Defaulting to automatic GPU Backend \
                           selection." maxlog=1
                end
            end
        end
    end

    @debug "Running automatic GPU backend selection..."
    for device in GPU_DEVICES
        if __is_loaded(device)
            @debug "Trying backend: $(_get_device_name(device))."
            if __is_functional(device)
                @debug "Using GPU backend: $(_get_device_name(device))."
                return device
            end
            @debug "GPU backend: $(_get_device_name(device)) is not functional."
        else
            @debug "Trigger package for backend ($(_get_device_name(device))): \
                    $(_get_trigger_pkgname(device)) not loaded."
        end
    end

    if force_gpu_usage
        throw(LuxDeviceSelectionException())
    else
        @warn """No functional GPU backend found! Defaulting to CPU.

                 1. If no GPU is available, nothing needs to be done.
                 2. If GPU is available, load the corresponding trigger package.
                     a. LuxCUDA.jl for NVIDIA CUDA Support.
                     b. LuxAMDGPU.jl for AMD GPU ROCM Support.
                     c. Metal.jl for Apple Metal GPU Support.""" maxlog=1
        return LuxCPUDevice
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

"""
    default_device_rng(::AbstractLuxDevice)

Returns the default RNG for the device. This can be used to directly generate parameters
and states on the device using
[WeightInitializers.jl](https://github.com/LuxDL/WeightInitializers.jl).
"""
function default_device_rng(D::AbstractLuxDevice)
    return error("""`default_device_rng` not implemented for $(typeof(D)). This is either because:

           1. The default RNG for this device is not known / officially provided.
           2. The trigger package for the device is not loaded.
           """)
end
default_device_rng(::LuxCPUDevice) = Random.default_rng()

# Dispatches for Different Data Structures
# Abstract Array / Tuples / NamedTuples have special fast paths to facilitate type stability
# For all other types we rely on fmap which means we lose type stability.
# For Lux, typically models only has these 3 datastructures so we should be mostly fine.
for (dev) in (:CPU, :CUDA, :AMDGPU, :Metal)
    ldev = Symbol("Lux$(dev)Device")
    ladaptor = Symbol("Lux$(dev)Adaptor")
    @eval begin
        function (D::$(ldev))(x::AbstractArray)
            fn = Base.Fix1(adapt, $(ladaptor)())
            return _isbitsarray(x) ? fn(x) : map(D, x)
        end
        (D::$(ldev))(x::Tuple) = map(D, x)
        (D::$(ldev))(x::NamedTuple{F}) where {F} = NamedTuple{F}(D(values(x)))
        function (::$(ldev))(x)
            _isleaf(x) && return adapt($(ladaptor)(), x)
            return fmap(Base.Fix1(adapt, $(ladaptor)()), x; exclude=_isleaf)
        end
        function (::$(ldev))(NN::LuxCore.AbstractExplicitLayer)
            @warn "Lux layers are stateless and hence don't participate in device \
                   transfers. Apply this function on the parameters and states generated \
                   using `Lux.setup`." maxlog=1
            return NN
        end
    end
end

# Query Device from Array
"""
    get_device(x::AbstractArray) -> AbstractLuxDevice

Returns the device of the array `x`. Trigger Packages must be loaded for this to return the
correct device.
"""
get_device(::AbstractArray) = LuxCPUDevice()

# Adapt Interface
abstract type AbstractLuxDeviceAdaptor end

struct LuxCPUAdaptor <: AbstractLuxDeviceAdaptor end
struct LuxCUDAAdaptor{ID} <: AbstractLuxDeviceAdaptor
    device_id::ID
end
struct LuxAMDGPUAdaptor{ID} <: AbstractLuxDeviceAdaptor
    device_id::ID
end
struct LuxMetalAdaptor <: AbstractLuxDeviceAdaptor end

adapt_storage(::LuxCPUAdaptor, x::AbstractRange) = x
adapt_storage(::LuxCPUAdaptor, x::AbstractArray) = adapt(Array, x)
adapt_storage(::LuxCPUAdaptor, rng::AbstractRNG) = rng

_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where {T} = isbitstype(T)
_isbitsarray(x) = false

_isleaf(::AbstractRNG) = true
_isleaf(x) = _isbitsarray(x) || Functors.isleaf(x)

# Chain Rules Core
function CRC.rrule(::typeof(adapt_storage), to::AbstractLuxDeviceAdaptor, x::AbstractArray)
    function ∇adapt_storage(Δ)
        dev = get_device(x)
        return (NoTangent(), NoTangent(), dev(Δ))
    end
    return adapt_storage(to, x), ∇adapt_storage
end

end

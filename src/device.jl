abstract type LuxAbstractDeviceAdaptor end

@generated function _get_device_name(t::T) where {T <: LuxAbstractDeviceAdaptor}
    return hasfield(T, :name) ? :(t.name) : :("")
end
@generated function _get_trigger_pkgid(t::T) where {T <: LuxAbstractDeviceAdaptor}
    return hasfield(T, :pkgid) ? :(t.pkgid) :
           :(PkgId(UUID("b2108857-7c20-44ae-9111-449ecde12c47"), "Lux"))
end

abstract type LuxAbstractGPUDeviceAdaptor <: LuxAbstractDeviceAdaptor end

struct LuxCPUAdaptor <: LuxAbstractDeviceAdaptor end
struct LuxAutoDetectGPUAdaptor <: LuxAbstractGPUDeviceAdaptor end
struct LuxNoGPUDetectedAdaptor <: LuxAbstractGPUDeviceAdaptor end
Base.@kwdef struct LuxCUDAAdaptor <: LuxAbstractGPUDeviceAdaptor
    name::String = "CUDA"
    pkgid::PkgId = PkgId(UUID("d0bbae9a-e099-4d5b-a835-1c6931763bda"), "LuxCUDA")
end
Base.@kwdef struct LuxAMDGPUAdaptor <: LuxAbstractGPUDeviceAdaptor
    name::String = "AMDGPU"
    pkgid::PkgId = PkgId(UUID("83120cb1-ca15-4f04-bf3b-6967d2e6b60b"), "LuxAMDGPU")
end

const GPU_BACKENDS = (LuxCUDAAdaptor(), LuxAMDGPUAdaptor())  # Order is important here

supported_gpu_backends() = map(_get_device_name, GPU_BACKENDS)

function _get_first_functional_gpu_backend(fail::Bool=false)
    for backend in GPU_BACKENDS
        if haskey(Base.loaded_modules, backend.pkgid)
            @debug "Trying backend: $(backend.name)."
            if getproperty(Base.loaded_modules[backend.pkgid], :functional)()
                @debug "Using GPU backend: $(backend.name)."
                return backend
            end
            @debug "GPU backend: $(backend.name) is not functional."
        else
            @debug """
            Trigger package for backend ($(backend.name)): $((backend.pkgid)) not loaded."""
        end
    end

    if fail
        @debug "No GPU backend is available / functional!"
        return LuxNoGPUDetectedAdaptor()
    else
        @debug "Lux will try to auto-detect GPU Backend later!"
        return LuxAutoDetectGPUAdaptor()
    end
end

function select_gpu_backend(args...)
    backend = @load_preference("gpu_backend", nothing)
    # If backend set with preferences, use it
    if backend !== nothing
        allowed_backends = supported_gpu_backends()
        idx = findfirst(isequal(backend), allowed_backends)
        if backend ∉ allowed_backends
            @warn """
            `gpu_backend` preference is set to $backend, which is not a valid backend.
            Valid backends are $allowed_backends).
            Defaulting to automatic GPU Backend selection.
            """ maxlog=1
        else
            @debug "Using GPU backend set in preferences: $backend."
            return GPU_BACKENDS[idx]
        end
    end

    @debug "Running automatic GPU backend selection..."
    return _get_first_functional_gpu_backend(args...)
end

const GPU_BACKEND = select_gpu_backend(false)

gpu_backend!(backend) = gpu_backend!(string(backend))
gpu_backend!(backend::LuxAbstractGPUDeviceAdaptor) = gpu_backend!(_get_device_name(backend))
gpu_backend!() = gpu_backend!("")
function gpu_backend!(backend::String)
    if backend == ""
        @delete_preferences!("gpu_backend")
        @info """Deleted the local preference for `gpu_backend`. Restart Julia to use the
        new backend."""
        return
    end

    allowed_backends = supported_gpu_backends()
    if backend == _get_device_name(GPU_BACKEND)
        @info "GPU backend is already set to $backend. No action is required."
        return
    end

    @assert backend in allowed_backends "`gpu_backend` must be one of $(allowed_backends)"

    @set_preferences!("gpu_backend"=>backend)
    @info "GPU backend has been set to $backend. Restart Julia to use the new backend."
    return
end

function set_backend_if_higher_priority!(backend::LuxAbstractGPUDeviceAdaptor)
    T = typeof(backend)
    if !(GPU_BACKEND isa T)
        if any(Base.Fix1(isa, GPU_BACKEND),
               (LuxAutoDetectGPUAdaptor, LuxNoGPUDetectedAdaptor))
            gpu_backend!(backend)
        else
            idx1 = findfirst(Base.Fix2(isa, T), GPU_BACKENDS)
            idx2 = findfirst(Base.Fix2(isa, typeof(GPU_BACKEND)), GPU_BACKENDS)
            idx2 > idx1 && gpu_backend!(backend)
        end
    end
end

# Device Transfer functions
function gpu end
function cpu end

gpu(x) = gpu(GPU_BACKEND, x)

## Show warning that trigger package might not have been loaded for preferences
function gpu(t::LuxAbstractGPUDeviceAdaptor, x)
    @warn """
    `gpu` function for device: $(_get_device_name(t)) has been called but the corresponding
    trigger package: $(_get_trigger_pkgid(t)) has not been loaded.
    """ maxlog=1
    return x
end

## Slow fallback with runtime detection of GPU backend
function gpu(::LuxAutoDetectGPUAdaptor, x)
    @warn """
    `gpu` function is being called, but no GPU was detected/functional while loading Lux.

    This could be due to:

        * GPUs not being functional.
        * Julia session was not restarted after the trigger packages were loaded for the
          first time.

    Start the julia session with `JULIA_DEBUG=Lux` and load `Lux` to get more information.
    Alternatively, you can set the logging level to `Debug` and run
    `Lux.select_gpu_backend()`. It is recommended to fix the issue before using `gpu`.

    Lux will try to auto-detect GPU backend at runtime. This operation is expensive.
    """ maxlog=1
    return gpu(_get_first_functional_gpu_backend(true), x)
end

## No GPU detected
function gpu(::LuxNoGPUDetectedAdaptor, x)
    @warn """
    No GPU has been detected or is functional.

    Defaulting back to the CPU. (No action is required if you want to run on the CPU).
    """ maxlog=1
    return x
end

_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where {T} = isbitstype(T)
_isbitsarray(x) = false

_isleaf(::AbstractRNG) = true
_isleaf(x) = _isbitsarray(x) || Functors.isleaf(x)

# adapt_storage(::LuxCUDAAdaptor, x) = CUDA.cu(x)
# adapt_storage(::LuxCUDAAdaptor, rng::AbstractRNG) = rng
# function adapt_storage(::LuxCPUAdaptor, x::CUDA.CUSPARSE.AbstractCuSparseMatrix)
#     return adapt(Array, x)
# end

adapt_storage(::LuxCPUAdaptor, x::AbstractRange) = x
adapt_storage(::LuxCPUAdaptor, x::AbstractArray) = adapt(Array, x)
adapt_storage(::LuxCPUAdaptor, rng::AbstractRNG) = rng

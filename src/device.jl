abstract type LuxDeviceAdaptor end

struct LuxCPUAdaptor <: LuxDeviceAdaptor end
struct LuxCUDAAdaptor <: LuxDeviceAdaptor end
struct LuxAMDGPUAdaptor <: LuxDeviceAdaptor end

# Order is important here
const GPU_BACKENDS = (:CUDA, :AMDGPU)
const GPU_BACKENDS_PKG_ID = (Base.PkgId(UUID("d0bbae9a-e099-4d5b-a835-1c6931763bda"),
                                        "LuxCUDA"),
                             Base.PkgId(UUID("83120cb1-ca15-4f04-bf3b-6967d2e6b60b"),
                                        "LuxAMDGPU"))
const GPU_BACKEND = Ref{Symbol}(:CUDA)

function _select_default_gpu_backend()
    backend = @load_preference("gpu_backend", nothing)
    # If backend set with preferences, use it
    if backend !== nothing
        if backend ∉ GPU_BACKENDS
            @warn """
            `gpu_backend` preference is set to $backend, which is not a valid backend.
            Valid backends are $(GPU_BACKENDS).
            Defaulting to automatic GPU Backend selection.
            """ maxlog=1
        else
            @debug "Using GPU backend set in preferences: $backend."
            return backend
        end
    end

    @debug "Running automatic GPU backend selection..."

    for (backend, pkgid) in zip(GPU_BACKENDS, GPU_BACKENDS_PKG_ID)
        if haskey(Base.loaded_modules, pkgid)
            @debug "Trying backend: $backend."
            if getproperty(Base.loaded_modules[pkgid], :functional)()
                @debug "Using GPU backend: $backend."
                return backend
            end
            @debug "GPU backend: $backend is not functional."
        end
    end

    # If nothing is loaded then choose CUDA
    return :CUDA
end

gpu_backend!(backend) = gpu_backend!(Symbol(backend))
function gpu_backend!(backend::Symbol)
    if backend == GPU_BACKEND
        @info """GPU backend is already set to $backend. No action is required."""
        return
    end

    @assert !(backend in GPU_BACKENDS) "`gpu_backend` must be one of $(GPU_BACKENDS)"

    @set_preferences!("gpu_backend"=>backend)
    @info """GPU backend has been set to $backend. Restart Julia to use the new backend."""
    return
end

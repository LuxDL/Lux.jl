abstract type LuxDeviceAdaptor end

struct LuxCPUAdaptor <: LuxDeviceAdaptor end
struct LuxCUDAAdaptor <: LuxDeviceAdaptor end
struct LuxAMDGPUAdaptor <: LuxDeviceAdaptor end

# const GPU_BACKENDS = ("CUDA", "AMD")
# const GPU_BACKEND = @load_preference("gpu_backend", "CUDA")

# """
#     gpu_backend!(backend::String)

# Set the `gpu_backend` for Lux.jl. By default it is set to `CUDA`.
# """
# function gpu_backend!(backend::String)
#     if backend == GPU_BACKEND
#         @info """
#         GPU backend is already set to: $backend.
#         No need to do anything else.
#         """
#         return
#     end

#     backend in GPU_BACKENDS || throw(ArgumentError("""
#     Unsupported GPU backend: $backend.
#     Supported backends are: $GPU_BACKENDS.
#     """))

#     @set_preferences!("gpu_backend" => backend)
#     @info """
#     New GPU backend set: $backend.
#     Restart your Julia session for this change to take effect!
#     """

#     return
# end

# adapt_storage(::LuxCUDAAdaptor, x) = CUDA.cu(x)
# adapt_storage(::LuxCUDAAdaptor, rng::AbstractRNG) = rng

# function adapt_storage(::LuxCPUAdaptor,
#     x::Union{AbstractRange, SparseArrays.AbstractSparseArray})
#     return x
# end
# adapt_storage(::LuxCPUAdaptor, x::AbstractArray) = adapt(Array, x)
# adapt_storage(::LuxCPUAdaptor, rng::AbstractRNG) = rng
# function adapt_storage(::LuxCPUAdaptor, x::CUDA.CUSPARSE.AbstractCuSparseMatrix)
#     return adapt(Array, x)
# end

_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where {T} = isbitstype(T)
_isbitsarray(x) = false

_isleaf(::AbstractRNG) = true
_isleaf(x) = _isbitsarray(x) || Functors.isleaf(x)

# """
#     cpu(x)

# Transfer `x` to CPU.
# """
# cpu(x) = fmap(x -> adapt(LuxCPUAdaptor(), x), x; exclude=_isleaf)

# """
#     gpu(x)

# Transfer `x` to GPU determined by the backend set using `Lux.gpu_backend!`.
# """
# function gpu(x)
#     @static if GPU_BACKEND == "CUDA"
#         gpu(LuxCUDAAdaptor(), x)
#     elseif GPU_BACKEND == "AMD"
#         gpu(LuxAMDAdaptor(), x)
#     else
#         error("""
#         Unsupported GPU backend: $GPU_BACKEND.
#         Supported backends are: $GPU_BACKENDS.
#         """)
#     end
#     # check_use_cuda()
#     # return use_cuda[] ? fmap(x -> adapt(LuxCUDAAdaptor(), x), x; exclude=_isleaf) : x
# end

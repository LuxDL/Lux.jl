abstract type LuxDeviceAdaptor end

struct LuxCPUAdaptor <: LuxDeviceAdaptor end
struct LuxCUDAAdaptor <: LuxDeviceAdaptor end

adapt_storage(::LuxCUDAAdaptor, x) = CUDA.cu(x)
adapt_storage(::LuxCUDAAdaptor, x::FillArrays.AbstractFill) = CUDA.cu(collect(x))
adapt_storage(::LuxCUDAAdaptor, x::Zygote.OneElement) = CUDA.cu(collect(x))
adapt_storage(::LuxCUDAAdaptor, rng::AbstractRNG) = rng

function adapt_storage(::LuxCPUAdaptor,
                       x::Union{AbstractRange, FillArrays.AbstractFill, Zygote.OneElement,
                                SparseArrays.AbstractSparseArray})
    return x
end
adapt_storage(::LuxCPUAdaptor, x::AbstractArray) = adapt(Array, x)
adapt_storage(::LuxCPUAdaptor, rng::AbstractRNG) = rng
# TODO(@avik-pal): SparseArrays
function adapt_storage(::LuxCPUAdaptor,
                       x::CUDA.CUSPARSE.CUDA.CUSPARSE.AbstractCuSparseMatrix)
    return adapt(Array, x)
end

_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where {T} = isbitstype(T)
_isbitsarray(x) = false

_isleaf(::AbstractRNG) = true
_isleaf(x) = _isbitsarray(x) || Functors.isleaf(x)

"""
    cpu(x)

Transfer `x` to CPU
"""
cpu(x) = fmap(x -> adapt(LuxCPUAdaptor(), x), x)

"""
    gpu(x)

Transfer `x` to GPU
"""
function gpu(x)
    check_use_cuda()
    return use_cuda[] ? fmap(x -> adapt(LuxCUDAAdaptor(), x), x; exclude=_isleaf) : x
end

function check_use_cuda()
    if use_cuda[] === nothing
        use_cuda[] = CUDA.functional()
        if use_cuda[] && !CUDA.has_cudnn()
            @warn """CUDA.jl found cuda, but did not find libcudnn. Some functionality
                     will not be available."""
        end
        if !(use_cuda[])
            @info """The GPU function is being called but the GPU is not accessible.
                     Defaulting back to the CPU. (No action is required if you want
                     to run on the CPU).""" maxlog=1
        end
    end
end

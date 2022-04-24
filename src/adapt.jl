abstract type EFLDeviceAdaptor end

struct EFLCPUAdaptor <: EFLDeviceAdaptor end
struct EFLCUDAAdaptor <: EFLDeviceAdaptor end

adapt_storage(::EFLCUDAAdaptor, x) = CUDA.cu(x)
adapt_storage(::EFLCUDAAdaptor, x::FillArrays.AbstractFill) = CUDA.cu(colelct(x))
adapt_storage(::EFLCUDAAdaptor, x::Zygote.OneElement) = CUDA.cu(collect(x))
adapt_storage(to::EFLCUDAAdaptor, x::ComponentArray) = ComponentArray(adapt_storage(to, getdata(x)), getaxes(x))
adapt_storage(::EFLCUDAAdaptor, rng::AbstractRNG) = rng

function adapt_storage(
    ::EFLCPUAdaptor,
    x::Union{AbstractArray,AbstractRange,FillArrays.AbstractFill,Zygote.OneElement,SparseArrays.AbstractSparseArray},
)
    return x
end
adapt_storage(to::EFLCPUAdaptor, x::ComponentArray) = ComponentArray(adapt_storage(to, getdata(x)), getaxes(x))
adapt_storage(::EFLCPUAdaptor, rng::AbstractRNG) = rng
# TODO: SparseArrays
adapt_storage(::EFLCPUAdaptor, x::CUDA.CUSPARSE.CUDA.CUSPARSE.AbstractCuSparseMatrix) = adapt(Array, x)

_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where {T} = isbitstype(T)
_isbitsarray(x) = false

_isleaf(::AbstractRNG) = true
_isleaf(x) = _isbitsarray(x) || Functors.isleaf(x)

cpu(x) = fmap(x -> adapt(EFLCPUAdaptor(), x), x)

function gpu(x)
    check_use_cuda()
    return use_cuda[] ? fmap(x -> adapt(EFLCUDAAdaptor(), x), x; exclude=_isleaf) : x
end

function check_use_cuda()
    if use_cuda[] === nothing
        use_cuda[] = CUDA.functional()
        if use_cuda[] && !CUDA.has_cudnn()
            @warn "CUDA.jl found cuda, but did not find libcudnn. Some functionality will not be available."
        end
        if !(use_cuda[])
            @info """The GPU function is being called but the GPU is not accessible. 
                     Defaulting back to the CPU. (No action is required if you want to run on the CPU).""" maxlog = 1
        end
    end
end

module LuxCUDAExt

using LuxCUDA, ..Lux
using Adapt, ChainRulesCore, Functors, Lux, NNlib, Random
using ..Lux: GPU_BACKEND, LuxCPUAdaptor, LuxCUDAAdaptor
import Adapt: adapt, adapt_storage
import ChainRulesCore as CRC

__init__() = LuxCUDA.functional() && Lux.set_backend_if_higher_priority!(LuxCUDAAdaptor())

# device.jl
function _check_cuda_gpu()
    if !LuxCUDA.functional()
        @warn """
        `gpu` function is called for CUDA but LuxCUDA.jl is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return nothing
end

CRC.@non_differentiable _check_cuda_gpu()
CRC.@non_differentiable LuxCUDA.functional()

function Lux.gpu(::LuxCUDAAdaptor, x)
    _check_cuda_gpu()
    return LuxCUDA.functional() ?
           fmap(x -> adapt(LuxCUDAAdaptor(), x), x; exclude=Lux._isleaf) : x
end

adapt_storage(::LuxCUDAAdaptor, x) = cu(x)
adapt_storage(::LuxCUDAAdaptor, rng::AbstractRNG) = rng

adapt_storage(::LuxCPUAdaptor, rng::CUDA.RNG) = Random.default_rng()
adapt_storage(::LuxCPUAdaptor, x::CUDA.CUSPARSE.AbstractCuSparseMatrix) = adapt(Array, x)

# utils.jl
Lux.replicate(rng::CUDA.RNG) = deepcopy(rng)

@inline function Lux._init_hidden_state(rng::AbstractRNG, rnn,
                                        x::Union{StridedCuArray, CuArray})
    return adapt(LuxCUDAAdaptor(), rnn.init_state(rng, rnn.out_dims, size(x, 2)))
end

@inline function Lux._conv(x::SubArray{T, N, <:CuArray}, weight, cdims) where {T, N}
    return conv(copy(x), weight, cdims)
end

@inline function Lux._conv_transpose(x::SubArray{T, N, <:CuArray}, weight,
                                     cdims) where {T, N}
    return ∇conv_data(copy(x), weight, cdims)
end

# chainrules.jl
CRC.rrule(::Type{Array}, x::CuArray) = Array(x), d -> (NoTangent(), cu(d))

function CRC.rrule(::typeof(adapt_storage), to::LuxCPUAdaptor,
                   x::Union{StridedCuArray, CuArray})
    return adapt_storage(to, x),
           d -> (NoTangent(), NoTangent(), adapt_storage(LuxCUDAAdaptor(), d))
end

function CRC.rrule(::typeof(adapt_storage), to::LuxCUDAAdaptor, x::Array)
    return adapt_storage(to, x),
           d -> (NoTangent(), NoTangent(), adapt_storage(LuxCPUAdaptor(), d))
end

end

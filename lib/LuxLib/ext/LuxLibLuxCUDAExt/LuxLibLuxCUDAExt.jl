module LuxLibLuxCUDAExt

using LuxCUDA, LuxLib
import ChainRulesCore as CRC
import LuxLib: batchnorm, batchnorm_cudnn, ∇batchnorm_cudnn, _get_batchnorm_statistics,
    FP_32_64, ∂∅

include("batchnorm.jl")

# utils.jl
LuxLib._replicate(rng::CUDA.RNG) = deepcopy(rng)

# api/batchnorm.jl
const CUDNN_BN_ARRAY_TYPE = Union{CuArray{<:FP_32_64, 2}, CuArray{<:FP_32_64, 4},
    CuArray{<:FP_32_64, 5}}
const BNParamType = Union{Nothing, CuVector{<:FP_32_64}}

function batchnorm(x::CUDNN_BN_ARRAY_TYPE, scale::BNParamType, bias::BNParamType,
    running_mean::BNParamType, running_var::BNParamType; momentum::Real, training::Val,
    epsilon::Real)
    rm, rv = _get_batchnorm_statistics(x, running_mean, running_var, training)

    x_ = first(batchnorm_cudnn(rm, rv, scale, bias, x, momentum, epsilon, training))
    return x_, (; running_mean=rm, running_var=rv)
end

function batchnorm_cudnn(running_mean, running_var, scale, bias, x, momentum, eps,
    training)
    return batchnorm_cudnn(scale, bias, x, running_mean, running_var, momentum,
        training; ϵ=eps)
end

function CRC.rrule(::typeof(batchnorm_cudnn), running_mean, running_var, scale, bias, x,
    momentum, epsilon, t::Val{training}) where {training}
    y, xmean, xivar = batchnorm_cudnn(running_mean, running_var, scale, bias, x, momentum,
        epsilon, t)
    function ∇batchnorm_cudnn_internal(Δ)
        ∂y = CRC.unthunk(first(Δ))
        ∂g, ∂b, ∂x = ∇batchnorm_cudnn(scale, bias, x, ∂y, running_mean, running_var, xmean,
            xivar; ϵ=epsilon)
        return (∂∅, ∂∅, ∂∅, ∂g, ∂b, ∂x, ∂∅, ∂∅, ∂∅)
    end
    return (y, xmean, xivar), ∇batchnorm_cudnn_internal
end

end

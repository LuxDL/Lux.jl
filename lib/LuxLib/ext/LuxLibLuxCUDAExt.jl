module LuxLibLuxCUDAExt

using LuxCUDA, LuxLib
import ChainRulesCore as CRC
import LuxLib: batchnorm, _batchnorm_cudnn!, _get_batchnorm_statistics, FP_32_64, ∂∅

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

    x_ = _batchnorm_cudnn!(rm, rv, scale, bias, x, momentum, epsilon, training)
    return x_, (; running_mean=rm, running_var=rv)
end

function _batchnorm_cudnn!(running_mean, running_var, scale, bias, x, momentum, eps,
    ::Val{training}) where {training}
    __batchnorm = @static if @isdefined(NNlibCUDA)
        NNlibCUDA.batchnorm
    else
        !hasproperty(NNlib, :batchnorm) &&
            throw(LuxLib.OutdatedNNlibDependencyException(:batchnorm))
        NNlib.batchnorm
    end
    return __batchnorm(scale, bias, x, running_mean, running_var, momentum; eps, training)
end

function CRC.rrule(::typeof(_batchnorm_cudnn!), running_mean, running_var, scale, bias, x,
    momentum, epsilon, t::Val{training}) where {training}
    y = _batchnorm_cudnn!(running_mean, running_var, scale, bias, x, momentum, epsilon, t)
    function ∇_batchnorm_cudnn!(Δ)
        __∇batchnorm = @static if @isdefined(NNlibCUDA)
            NNlibCUDA.∇batchnorm
        else
            !hasproperty(NNlib, :∇batchnorm) &&
                throw(LuxLib.OutdatedNNlibDependencyException(:∇batchnorm))
            NNlib.∇batchnorm
        end
        ∂g, ∂b, ∂x = __∇batchnorm(scale, bias, x, CRC.unthunk(Δ), running_mean, running_var,
            momentum; eps=epsilon, training)
        return (∂∅, ∂∅, ∂∅, ∂g, ∂b, ∂x, ∂∅, ∂∅, ∂∅)
    end
    return y, ∇_batchnorm_cudnn!
end

end

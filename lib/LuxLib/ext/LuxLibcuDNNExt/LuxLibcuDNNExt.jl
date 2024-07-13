module LuxLibcuDNNExt

using LuxLib: LuxLib, Optional
using CUDA: CUDA, CuArray, CuVector, CU_NULL, DenseCuArray
using ChainRulesCore: ChainRulesCore
using cuDNN: cuDNN, CUDNN_BN_MIN_EPSILON, cudnnBatchNormalizationBackward,
             cudnnBatchNormalizationForwardInference, CUDNN_BATCHNORM_SPATIAL,
             cudnnBatchNormalizationForwardTraining, cudnnTensorDescriptor,
             CUDNN_TENSOR_NCHW, cudnnDataType
using FastClosures: @closure

const CRC = ChainRulesCore

include("batchnorm.jl")

# api/batchnorm.jl
const CUDNN_BN_ARRAY_TYPE = Union{
    CuArray{<:Union{Float32, Float64}, 2}, CuArray{<:Union{Float32, Float64}, 4},
    CuArray{<:Union{Float32, Float64}, 5}}
const BNParamType = Optional{<:CuVector{<:Union{Float32, Float64}}}

function LuxLib.batchnorm(x::CUDNN_BN_ARRAY_TYPE, scale::BNParamType, bias::BNParamType,
        running_mean::BNParamType, running_var::BNParamType, training::Val,
        σ::F=identity, momentum::Real=0.1f0, epsilon::Real=1.0f-5) where {F}
    rm, rv = LuxLib._get_batchnorm_statistics(x, running_mean, running_var, training)
    x_ = LuxLib.batchnorm_cudnn(rm, rv, scale, bias, x, momentum, epsilon, training)[1]
    return LuxLib.fast_activation!!(σ, x_), (; running_mean=rm, running_var=rv)
end

function LuxLib.batchnorm_cudnn(
        running_mean, running_var, scale, bias, x, momentum, eps, training)
    return LuxLib.batchnorm_cudnn(
        scale, bias, x, running_mean, running_var, momentum, training; ϵ=eps)
end

function CRC.rrule(::typeof(LuxLib.batchnorm_cudnn), running_mean, running_var, scale,
        bias, x, momentum, epsilon, t::Val{training}) where {training}
    # TODO: Transition this to an error in the future
    !training && @warn "`training=Val(false)` but gradient was called." maxlog=1
    y, xmean, xivar = LuxLib.batchnorm_cudnn(
        running_mean, running_var, scale, bias, x, momentum, epsilon, t)
    ∇batchnorm_cudnn_internal = @closure Δ -> begin
        ∂y = CRC.unthunk(first(Δ))
        ∂g, ∂b, ∂x = LuxLib.∇batchnorm_cudnn(
            scale, bias, x, ∂y, running_mean, running_var, xmean, xivar; ϵ=epsilon)
        return (CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), ∂g, ∂b,
            ∂x, CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent())
    end
    return (y, xmean, xivar), ∇batchnorm_cudnn_internal
end

end

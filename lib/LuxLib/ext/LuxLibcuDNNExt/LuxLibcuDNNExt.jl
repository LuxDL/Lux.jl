module LuxLibcuDNNExt

using LuxLib: LuxLib, Optional, ∂∅
using CUDA: CUDA, CuArray, CuVector, CU_NULL, DenseCuArray
using ChainRulesCore: ChainRulesCore
using cuDNN: cuDNN, cudnnBatchNormalizationBackward,
             cudnnBatchNormalizationForwardInference, CUDNN_BATCHNORM_SPATIAL,
             cudnnBatchNormalizationForwardTraining, cudnnTensorDescriptor,
             CUDNN_TENSOR_NCHW, cudnnDataType
using FastClosures: @closure
using Static: StaticBool, known, static

const CRC = ChainRulesCore

const CUDNNFloat = Union{Float32, Float64}

include("batchnorm.jl")

# api/batchnorm.jl
const CUDNN_BN_ARRAY_TYPE = Union{
    CuArray{<:CUDNNFloat, 2}, CuArray{<:CUDNNFloat, 4}, CuArray{<:CUDNNFloat, 5}}
const BNParamType = Optional{<:CuVector{<:CUDNNFloat}}

function LuxLib.batchnorm(x::CUDNN_BN_ARRAY_TYPE, scale::BNParamType, bias::BNParamType,
        running_mean::BNParamType, running_var::BNParamType,
        training::Union{Val, StaticBool}, σ::F=identity,
        momentum::Real=0.1f0, epsilon::Real=1.0f-5) where {F}
    rm, rv = LuxLib._get_batchnorm_statistics(
        x, running_mean, running_var, static(training))
    x_ = LuxLib.batchnorm_cudnn(
        rm, rv, scale, bias, x, momentum, epsilon, static(training))[1]
    return LuxLib.fast_activation!!(σ, x_), (; running_mean=rm, running_var=rv)
end

function LuxLib.batchnorm_cudnn(
        running_mean, running_var, scale, bias, x, momentum, eps, training)
    return LuxLib.batchnorm_cudnn(
        scale, bias, x, running_mean, running_var, momentum, training; ϵ=eps)
end

function CRC.rrule(::typeof(LuxLib.batchnorm_cudnn), running_mean, running_var,
        scale, bias, x, momentum, epsilon, training::StaticBool)
    # TODO: Transition this to an error in the future
    known(training) || @warn "`training=Val(false)` but gradient was called." maxlog=1
    y, xmean, xivar = LuxLib.batchnorm_cudnn(
        running_mean, running_var, scale, bias, x, momentum, epsilon, training)
    proj_g = CRC.ProjectTo(scale)
    proj_b = CRC.ProjectTo(bias)
    proj_x = CRC.ProjectTo(x)
    ∇batchnorm_cudnn_internal = @closure Δ -> begin
        ∂g, ∂b, ∂x = LuxLib.∇batchnorm_cudnn(scale, bias, x, CRC.unthunk(first(Δ)),
            running_mean, running_var, xmean, xivar; ϵ=epsilon)
        return ∂∅, ∂∅, ∂∅, proj_g(∂g), proj_b(∂b), proj_x(∂x), ∂∅, ∂∅, ∂∅
    end
    return (y, xmean, xivar), ∇batchnorm_cudnn_internal
end

end

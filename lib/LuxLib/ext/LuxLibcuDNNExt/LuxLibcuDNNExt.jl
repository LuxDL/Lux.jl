module LuxLibcuDNNExt

using LuxLib: LuxLib, Optional, ∂∅, Impl, Utils
using CUDA: CUDA, CuArray, CuVector, CU_NULL, DenseCuArray, DenseCuVector
using ChainRulesCore: ChainRulesCore
using cuDNN: cuDNN, cudnnBatchNormalizationBackward,
             cudnnBatchNormalizationForwardInference, CUDNN_BATCHNORM_SPATIAL,
             cudnnBatchNormalizationForwardTraining, cudnnTensorDescriptor,
             CUDNN_TENSOR_NCHW, cudnnDataType
using FastClosures: @closure
using Static: StaticBool

const CRC = ChainRulesCore

const cuDNNFloat = Union{Float32, Float64}

include("batchnorm.jl")

# api/batchnorm.jl
const CUDNN_BN_ARRAY_TYPE = Union{
    CuArray{<:cuDNNFloat, 2}, CuArray{<:cuDNNFloat, 4}, CuArray{<:cuDNNFloat, 5}}
const BNParamType = Optional{<:CuVector{<:cuDNNFloat}}

function Impl.batchnorm(
        x::CUDNN_BN_ARRAY_TYPE, γ::BNParamType, β::BNParamType, rμ::BNParamType,
        rσ²::BNParamType, training::StaticBool, σ::F, m::Real, ϵ::Real) where {F}
    rμₙ, rσ²ₙ = Impl.get_batchnorm_statistics(x, rμ, rσ², training)
    y = Impl.batchnorm_cudnn(γ, β, x, rμₙ, rσ²ₙ, m, ϵ, training)[1]
    return Impl.activation!!(σ, y), vec(rμₙ), vec(rσ²ₙ)
end

function CRC.rrule(
        ::typeof(Impl.batchnorm_cudnn), γ, β, x, rμ, rσ², m, ϵ, training::StaticBool)
    # TODO: Transition this to an error in the future
    Utils.known(training) || @warn "`training=Val(false)` but gradient was called." maxlog=1
    y, xμ, xσ⁻² = Impl.batchnorm_cudnn(γ, β, x, rμ, rσ², m, ϵ, training)
    𝒫x, 𝒫γ, 𝒫β = CRC.ProjectTo(x), CRC.ProjectTo(γ), CRC.ProjectTo(β)
    ∇batchnorm_cudnn = @closure Δ -> begin
        ∂γ, ∂β, ∂x = Impl.∇batchnorm_cudnn(
            γ, β, x, CRC.unthunk(first(Δ)), rμ, rσ², xμ, xσ⁻², ϵ)
        return ∂∅, 𝒫γ(∂γ), 𝒫β(∂β), 𝒫x(∂x), ∂∅, ∂∅, ∂∅, ∂∅, ∂∅
    end
    return (y, xμ, xσ⁻²), ∇batchnorm_cudnn
end

end

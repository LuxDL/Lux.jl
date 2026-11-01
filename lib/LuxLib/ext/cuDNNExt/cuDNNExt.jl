module cuDNNExt

using LuxLib: LuxLib, Optional, ∂∅, Impl
using LuxLib.Utils: Utils, safe_reshape, safe_vec, unsafe_known, recursive_unthunk
using CUDA: CUDA, CuArray, CuVector, DenseCuArray, DenseCuVector
using ChainRulesCore: ChainRulesCore
using cuDNN: batchnorm_gradient!, batchnorm_gradient_supported,
             batchnorm_inference!, batchnorm_inference_supported,
             batchnorm_training!, batchnorm_training_supported
using FastClosures: @closure
using Static: StaticBool, False, True

const CRC = ChainRulesCore

const cuDNNFloat = Union{Float32,Float64}

const OptionalVector = Union{Nothing,AbstractVector}
const BatchNormFallbackSignature =
    Tuple{AbstractArray,OptionalVector,OptionalVector,OptionalVector,OptionalVector,
          StaticBool,Any,Any,Any}

function batchnorm_fallback(x, γ, β, rμ, rσ², training, σ, momentum, epsilon)
    return invoke(Impl.batchnorm, BatchNormFallbackSignature, x, γ, β, rμ, rσ²,
                  training, σ, momentum, epsilon)
end

include("batchnorm.jl")

function batchnorm_cudnn_supported(γ, β, x, rμ, rσ², training, momentum, epsilon)
    γ === nothing && return false
    β === nothing && return false
    if ndims(x) == 2
        x = reshape(x, 1, 1, size(x, 1), size(x, 2))
    end
    dims = wsize(x, True())
    γ = reshape(γ, dims)
    β = reshape(β, dims)
    rμ = safe_reshape(rμ, dims...)
    rσ² = safe_reshape(rσ², dims...)
    if unsafe_known(training)
        batchnorm_training_supported(x, x, γ, β; running_mean=rμ,
                                     running_var=rσ², momentum,
                                     epsilon) || return false
        return batchnorm_gradient_supported(x, γ, β, x, x, γ, γ, γ)
    end
    return batchnorm_inference_supported(x, x, γ, β, rμ, rσ²)
end

CRC.@non_differentiable batchnorm_cudnn_supported(::Any...)

# api/batchnorm.jl
function Impl.batchnorm(
    x::Union{<:CuArray{T,2},<:CuArray{T,4},<:CuArray{T,5}},
    γ::Optional{<:CuVector{T}},
    β::Optional{<:CuVector{T}},
    rμ::Optional{<:CuVector{T}},
    rσ²::Optional{<:CuVector{T}},
    training::StaticBool,
    σ::F,
    m,
    ϵ,
) where {T<:cuDNNFloat,F}
    rμₙ, rσ²ₙ = Impl.get_batchnorm_statistics(x, rμ, rσ², training)
    if batchnorm_cudnn_supported(γ, β, x, rμₙ, rσ²ₙ, training, m, ϵ)
        y = Impl.batchnorm_cudnn(γ, β, x, rμₙ, rσ²ₙ, m, ϵ, training)[1]
        return Impl.activation!!(σ, y), safe_vec(rμₙ), safe_vec(rσ²ₙ)
    end
    return batchnorm_fallback(x, γ, β, rμ, rσ², training, σ, m, ϵ)
end

function CRC.rrule(
    ::typeof(Impl.batchnorm_cudnn), γ, β, x, rμ, rσ², m, ϵ, training::StaticBool
)
    y, xμ, xσ⁻² = Impl.batchnorm_cudnn(γ, β, x, rμ, rσ², m, ϵ, training)
    𝒫x, 𝒫γ, 𝒫β = CRC.ProjectTo(x), CRC.ProjectTo(γ), CRC.ProjectTo(β)
    ∇batchnorm_cudnn = @closure Δ -> begin
        ∂γ, ∂β, ∂x = Impl.∇batchnorm_cudnn(
            γ, β, x, recursive_unthunk(first(Δ)), rμ, rσ², xμ, xσ⁻², ϵ
        )
        return ∂∅, 𝒫γ(∂γ), 𝒫β(∂β), 𝒫x(∂x), ∂∅, ∂∅, ∂∅, ∂∅, ∂∅
    end
    return (y, xμ, xσ⁻²), ∇batchnorm_cudnn
end

end

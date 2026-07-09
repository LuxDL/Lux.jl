module cuDNNExt

using LuxLib: LuxLib, Optional, ∂∅, Impl
using LuxLib.Utils: Utils, safe_reshape, safe_vec, unsafe_known, recursive_unthunk
using CUDA: CUDA, CuArray, CuVector, DenseCuArray, DenseCuVector
using ChainRulesCore: ChainRulesCore
using cuDNN: batchnorm_gradient!, batchnorm_inference!, batchnorm_training!
using FastClosures: @closure
using Static: StaticBool, False, True

const CRC = ChainRulesCore

const cuDNNFloat = Union{Float32,Float64}

include("batchnorm.jl")

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
    y = Impl.batchnorm_cudnn(γ, β, x, rμₙ, rσ²ₙ, m, ϵ, training)[1]
    return Impl.activation!!(σ, y), safe_vec(rμₙ), safe_vec(rσ²ₙ)
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

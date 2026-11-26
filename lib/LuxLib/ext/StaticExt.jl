module StaticExt

using ChainRulesCore: ChainRulesCore
using LuxLib: LuxLib, API, Impl, Optional, Utils, ∂∅
using Random: AbstractRNG
using Static: StaticBool, StaticInt, StaticSymbol

Utils.static(::StaticBool{B}) where {B} = Val(B)
Utils.static(::StaticInt{N}) where {N} = Val(N)
Utils.static(::StaticSymbol{S}) where {S} = Val(S)

Utils.known(::StaticBool{B}) where {B} = B
Utils.known(::StaticInt{N}) where {N} = N
Utils.known(::StaticSymbol{S}) where {S} = S
Utils.known(::Type{<:StaticBool{B}}) where {B} = B
Utils.known(::Type{<:StaticInt{N}}) where {N} = N
Utils.known(::Type{<:StaticSymbol{S}}) where {S} = S

function Utils.static_training_mode(training::StaticBool, args...)
    return Utils.static_training_mode_check(
        training, Utils.static(training), Utils.within_autodiff_vararg(args...)
    )
end

function ChainRulesCore.rrule(
    ::typeof(Utils.static_training_mode), training::StaticBool, args...
)
    res = Utils.static_training_mode_check(training, Utils.static(training), Utils.True())
    return res, _ -> ntuple(Returns(∂∅), length(args) + 2)
end

function API.batchnorm(
    x::AbstractArray,
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    rμ::Optional{<:AbstractVector},
    rσ²::Optional{<:AbstractVector},
    training::StaticBool,
    σ::F=identity,
    momentum::Real=0.1f0,
    epsilon::Real=Utils.default_epsilon(x),
) where {F}
    return API.batchnorm(x, γ, β, rμ, rσ², Utils.static(training), σ, momentum, epsilon)
end

function API.instancenorm(
    x::AbstractArray,
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    training::StaticBool,
    σ::F=identity,
    epsilon=Utils.default_epsilon(x),
) where {F}
    return API.instancenorm(x, γ, β, Utils.static(training), σ, epsilon)
end

function API.instancenorm(
    x::AbstractArray,
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    rμ::Optional{<:AbstractVector},
    rσ²::Optional{<:AbstractVector},
    training::StaticBool,
    σ::F=identity,
    momentum::Optional{<:Number}=0.1f0,
    epsilon=Utils.default_epsilon(x),
) where {F}
    return API.instancenorm(x, γ, β, rμ, rσ², Utils.static(training), σ, momentum, epsilon)
end

function API.dropout(
    rng::AbstractRNG, x::AbstractArray, p::T, training::StaticBool, invp::T, dims
) where {T}
    return API.dropout(rng, x, p, Utils.static(training), invp, dims)
end

function API.dropout(
    rng::AbstractRNG,
    x::AbstractArray,
    mask::AbstractArray,
    p::T,
    training::StaticBool,
    update_mask::Union{API.TrainingType,StaticBool},
    invp::T,
    dims,
) where {T}
    return API.dropout(
        rng, x, mask, p, Utils.static(training), Utils.static(update_mask), invp, dims
    )
end

function API.dropout(
    rng::AbstractRNG,
    x::AbstractArray,
    mask::AbstractArray,
    p::T,
    training::API.TrainingType,
    update_mask::StaticBool,
    invp::T,
    dims,
) where {T}
    return API.dropout(rng, x, mask, p, training, Utils.static(update_mask), invp, dims)
end

function API.alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, training::StaticBool)
    return API.alpha_dropout(rng, x, p, Utils.static(training))
end

function API.alpha_dropout(
    rng::AbstractRNG, x::AbstractArray, p, training::StaticBool, α, A, B
)
    return API.alpha_dropout(rng, x, p, Utils.static(training), α, A, B)
end

function Impl.normalization(
    x::AbstractArray,
    rμ::Optional{<:AbstractVector},
    rσ²::Optional{<:AbstractVector},
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    reduce_dims,
    training::StaticBool,
    momentum,
    epsilon,
    act::F=identity,
) where {F}
    return Impl.normalization(
        x, rμ, rσ², γ, β, reduce_dims, Utils.static(training), momentum, epsilon, act
    )
end

end

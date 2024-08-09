# Deprecations for version 1.0
import .API: batchnorm, groupnorm, instancenorm, layernorm, dropout,
             fused_conv_bias_activation

## normalization
@deprecate batchnorm(x, scale, bias, running_mean, running_var, σ::F=identity;
    momentum::Real, training::Val, epsilon::Real) where {F} batchnorm(
    x, scale, bias, running_mean, running_var, training, σ, momentum, epsilon)

@deprecate groupnorm(x, scale, bias, σ::F=identity; groups::Int, epsilon::Real) where {F} groupnorm(
    x, scale, bias, groups, σ, epsilon)

@deprecate instancenorm(x, scale, bias, σ::F=identity; epsilon, training) where {F} instancenorm(
    x, scale, bias, training, σ, epsilon)

@deprecate layernorm(x, scale, bias, σ::F=identity; dims, epsilon) where {F} layernorm(
    x, scale, bias, σ, dims, epsilon)

## dropout
@deprecate dropout(
    rng::AbstractRNG, x::AbstractArray, p::T, training::Val, invp::T; dims) where {T} dropout(
    rng, x, p, training, invp, dims)

@deprecate dropout(
    rng::AbstractRNG, x::AbstractArray, p::T, training::Val; dims, invp::T=inv(p)) where {T} dropout(
    rng, x, p, training, invp, dims)

@deprecate dropout(rng::AbstractRNG, x::AbstractArray{T1, N}, mask::AbstractArray{T2, N},
    p::T, training::Val, um::Val, invp::T; dims) where {T, T1, T2, N} dropout(
    rng, x, mask, p, training, um, invp, dims)

@deprecate dropout(rng::AbstractRNG, x::AbstractArray{T1, N}, mask::AbstractArray{T2, N},
    p::T, training::Val, um::Val; dims, invp::T=inv(p)) where {T, T1, T2, N} dropout(
    rng, x, mask, p, training, um, invp, dims)

## conv
@deprecate fused_conv_bias_activation(
    σ::F, weight::AbstractArray{<:Number, N}, x::AbstractArray{<:Number, N},
    b::AbstractArray{<:Number, N}, cdims::ConvDims) where {F, N} fused_conv_bias_activation(
    σ, weight, x, _vec(b), cdims)

## Private API that was at a point being illegally used in Lux
@deprecate __∇conv_data(args...; kwargs...) Impl.∇conv_data(args...; kwargs...)

@deprecate __apply_bias_activation(σ::F, x, bias::AbstractArray) where {F} bias_activation(
    σ, x, _vec(bias))

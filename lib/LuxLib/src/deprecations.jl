Base.@deprecate batchnorm(
    x::AbstractArray{<:Real, N}, scale::Union{Nothing, <:AbstractVector},
    bias::Union{Nothing, <:AbstractVector}, running_mean::Union{Nothing, <:AbstractVector},
    running_var::Union{Nothing, <:AbstractVector}, σ::F=identity;
    momentum::Real, training::Val, epsilon::Real) where {F, N} batchnorm(
    x, scale, bias, running_mean, running_var, training, σ, momentum,  epsilon)

Base.@deprecate instancenorm(
    x::AbstractArray{<:Real, N}, scale::Union{Nothing, <:AbstractVector},
    bias::Union{Nothing, <:AbstractVector}, σ::F=identity;
    training::Val, epsilon::Real=1f-5) where {F, N} instancenorm(
    x, scale, bias, training, σ, epsilon)

Base.@deprecate layernorm(
    x::AbstractArray{<:Real, N}, scale::Union{Nothing, <:AbstractVector},
    bias::Union{Nothing, <:AbstractVector}, σ::F=identity;
    dims=Colon(), epsilon::Real=1f-5) where {F, N} layernorm(
    x, scale, bias, σ, dims, epsilon)

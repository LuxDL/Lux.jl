Base.@deprecate batchnorm(
    x::AbstractArray{<:Real, N}, scale::Union{Nothing, <:AbstractVector},
    bias::Union{Nothing, <:AbstractVector}, running_mean::Union{Nothing, <:AbstractVector},
    running_var::Union{Nothing, <:AbstractVector}, σ::F=identity;
    momentum::Real, training::Val, epsilon::Real) where {F, N} batchnorm(
    x, scale, bias, running_mean, running_var, training, σ, momentum,  epsilon)

# Deprecations for version 0.4
@deprecate batchnorm(x, scale, bias, running_mean, running_var, σ::F=identity;
    momentum::Real, training::Val, epsilon::Real) where {F} batchnorm(
    x, scale, bias, running_mean, running_var, training, σ, momentum, epsilon)

@deprecate groupnorm(x, scale, bias, σ::F=identity; groups::Int, epsilon::Real) where {F} groupnorm(
    x, scale, bias, groups, σ, epsilon)

@deprecate instancenorm(x, scale, bias, σ::F=identity; epsilon, training) where {F} instancenorm(
    x, scale, bias, training, σ, epsilon)

@deprecate layernorm(x, scale, bias, σ::F=identity; dims, epsilon) where {F} layernorm(
    x, scale, bias, σ, dims, epsilon)

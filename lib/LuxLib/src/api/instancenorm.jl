@doc doc"""
    instancenorm(x, scale, bias, training, act, epsilon = eps(eltype(x)) ^ (5 // 7))
    instancenorm(x, scale, bias, running_mean, running_var, training, act, momentum,
        epsilon = eps(eltype(x)) ^ (5 // 7))

Instance Normalization. For details see [1].

Instance Normalization computes the mean and variance for each
``D_1 \times ... \times D_{N - 2} \times 1 \times 1`` input slice and normalises the input
accordingly.

## Arguments

  - `x`: Input to be Normalized (must be atleast 3D)
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)
  - `running_mean`: Running mean (can be `nothing`)
  - `running_var`: Running variance (can be `nothing`)
  - `training`: Set to `Val(true)` or `True()` if running in training mode. Can be set to
    `nothing` to automatically determine if the function is being called within an autodiff
     context
  - `σ`: Activation function (default: `identity`)
  - `epsilon`: Value added to the denominator for numerical stability
    (default: `eps(eltype(x)) ^ (5 / 7)`)
  - `momentum`: Momentum for updating running mean and variance (default: `0.1f0`)

## Returns

Normalized Array of same size as `x`. And a Named Tuple containing the updated running
mean and variance.

## References

[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The
    missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).
"""
function instancenorm(x::AbstractArray, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, training::TrainingType,
        σ::F=identity, epsilon::Real=default_epsilon(x)) where {F}
    # This API is kept for legacy purposes when we didn't support passing running stats
    return instancenorm(x, γ, β, nothing, nothing, training, σ, nothing, epsilon)
end

function instancenorm(x::AbstractArray, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, rμ::Optional{<:AbstractVector},
        rσ²::Optional{<:AbstractVector}, training::TrainingType, σ::F=identity,
        momentum::Optional{<:Real}=0.1f0, epsilon::Real=default_epsilon(x)) where {F}
    assert_valid_instancenorm_arguments(x)

    y, rμₙ, rσ²ₙ = instancenorm_impl(
        x, γ, β, rμ, rσ², static_training_mode(training, x, γ, β, rμ, rσ²),
        select_fastest_activation(σ, x, γ, β), momentum, epsilon)

    return y, (; running_mean=remove_tracking(rμₙ), running_var=remove_tracking(rσ²ₙ))
end

function assert_valid_instancenorm_arguments(::AbstractArray{T, N}) where {T, N}
    @assert N>2 "`ndims(x) = $(N)` must be at least > 2."
    return nothing
end

CRC.@non_differentiable assert_valid_instancenorm_arguments(::Any...)

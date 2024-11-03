@doc doc"""
    batchnorm(x, scale, bias, running_mean, running_var, training,
        σ=identity, momentum = 0.1f0, epsilon = eps(eltype(x)) ^ (5 // 7))

Batch Normalization. For details see [1].

Batch Normalization computes the mean and variance for each
``D_1 \times ... \times D_{N - 2} \times 1 \times D_N`` input slice and normalises the input
accordingly.

## Arguments

  - `x`: Input to be Normalized
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)
  - `running_mean`: Running mean (can be `nothing`)
  - `running_var`: Running variance (can be `nothing`)
  - `training`: Set to `Val(true)` or `True()` if running in training mode. Can be set to
    `nothing` to automatically determine if the function is being called within an autodiff
     context
  - `σ`: Activation function (default: `identity`)
  - `momentum`: Momentum for updating running mean and variance (default: `0.1f0`)
  - `epsilon`: Value added to the denominator for numerical stability
    (default: `eps(eltype(x)) ^ (5 / 7)`)

## Returns

Normalized Array of same size as `x`. And a Named Tuple containing the updated running
mean and variance.

## References

[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network
    training by reducing internal covariate shift." International conference on machine
    learning. PMLR, 2015.
"""
function batchnorm(x::AbstractArray{T, N}, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, rμ::Optional{<:AbstractVector},
        rσ²::Optional{<:AbstractVector}, training::TrainingType, act::F=identity,
        momentum::Real=0.1f0, epsilon::Real=default_epsilon(x)) where {F, T, N}
    σ = select_fastest_activation(act, x, γ, β, rμ, rσ²)
    y, rμ, rσ² = batchnorm_impl(
        x, γ, β, rμ, rσ², static_training_mode(training, x, γ, β, rμ, rσ²),
        σ, momentum, epsilon)
    return y, (; running_mean=remove_tracking(rμ), running_var=remove_tracking(rσ²))
end

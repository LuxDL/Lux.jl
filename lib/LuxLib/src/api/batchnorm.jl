@doc doc"""
    batchnorm(x, scale, bias, running_mean, running_var, training::Union{Val, StaticBool},
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
  - `training`: Set to `Val(true)` if running in training mode
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
        rσ²::Optional{<:AbstractVector}, training::Union{Val, StaticBool},
        act::F=identity, momentum::Real=0.1f0,
        epsilon::Real=get_utils(:default_epsilon)(x)) where {F, T, N}
    σ = get_impl(:select_fastest_activation)(act, x, γ, β, rμ, rσ²)
    y, rμ, rσ² = get_impl(:batchnorm)(
        x, γ, β, rμ, rσ², static(training), σ, momentum, epsilon)
    return (y,
        (; running_mean=get_utils(:remove_tracking)(rμ),
            running_var=get_utils(:remove_tracking)(rσ²)))
end

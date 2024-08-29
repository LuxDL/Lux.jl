@doc doc"""
    instancenorm(x, scale, bias, training, σ = identity,
        epsilon = eps(eltype(x)) ^ (5 // 7))

Instance Normalization. For details see [1].

Instance Normalization computes the mean and variance for each
``D_1 \times ... \times D_{N - 2} \times 1 \times 1`` input slice and normalises the input
accordingly.

## Arguments

  - `x`: Input to be Normalized (must be atleast 3D)
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)
  - `σ`: Activation function (default: `identity`)
  - `epsilon`: Value added to the denominator for numerical stability
    (default: `eps(eltype(x)) ^ (5 / 7)`)
  - `training`: Set to `Val(true)` or `True()` if running in training mode. Can be set to
    `nothing` to automatically determine if the function is being called within an autodiff
     context

## Returns

Normalized Array of same size as `x`. And a Named Tuple containing the updated running
mean and variance.

## References

[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The
    missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).
"""
function instancenorm(x::AbstractArray, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, training::TrainingType,
        σ::F=identity, epsilon::Real=default_epsilon(x)) where {F}
    assert_valid_instancenorm_arguments(x)

    y, xμ, xσ² = instancenorm_impl(x, nothing, nothing, scale, bias,
        static_training_mode(training, x, scale, bias), nothing, epsilon,
        select_fastest_activation(σ, x, scale, bias))

    return y, (; running_mean=xμ, running_var=xσ²)
end

function assert_valid_instancenorm_arguments(::AbstractArray{T, N}) where {T, N}
    @assert N>2 "`ndims(x) = $(N)` must be at least > 2."
    return nothing
end

CRC.@non_differentiable assert_valid_instancenorm_arguments(::Any...)

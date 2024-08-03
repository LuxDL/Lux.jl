@doc doc"""
    instancenorm(x, scale, bias, training::Union{Val, StaticBool}, σ = identity,
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
  - `training`: Set to `Val(true)` if running in training mode

## Returns

Normalized Array of same size as `x`. And a Named Tuple containing the updated running
mean and variance.

## References

[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The
    missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).
"""
function instancenorm(x::AbstractArray{<:Real, N}, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, training::Union{Val, StaticBool},
        σ::F=identity, epsilon::Real=__default_epsilon(x)) where {N, F}
    _test_valid_instancenorm_arguments(x)

    x_, xm, xv = _normalization(
        x, nothing, nothing, scale, bias, _get_instancenorm_reduce_dims(x),
        static(training), nothing, epsilon, select_fastest_activation(σ, x, scale, bias))

    return x_, (; running_mean=xm, running_var=xv)
end

@generated function _get_instancenorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return :($(static.(Tuple([1:(N - 2)]...))))
end

function _test_valid_instancenorm_arguments(::AbstractArray{T, N}) where {T, N}
    N > 2 || throw(ArgumentError("`ndims(x) = $(N)` must be at least > 2."))
    return nothing
end

CRC.@non_differentiable _test_valid_instancenorm_arguments(::Any...)

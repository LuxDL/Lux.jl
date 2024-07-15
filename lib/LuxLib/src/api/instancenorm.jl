@doc doc"""
    instancenorm(x, scale, bias, training::Val, σ = identity, epsilon = 1f-5)

Instance Normalization. For details see [1].

Instance Normalization computes the mean and variance for each
``D_1 \times ... \times D_{N - 2} \times 1 \times 1`` input slice and normalises the input
accordingly.

## Arguments

  - `x`: Input to be Normalized (must be atleast 3D)
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)
  - `σ`: Activation function (default: `identity`)
  - `epsilon`: Value added to the denominator for numerical stability (default: `1f-5`)
  - `training`: Set to `Val(true)` if running in training mode

## Returns

Normalized Array of same size as `x`. And a Named Tuple containing the updated running
mean and variance.

## References

[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The
    missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).
"""
@stable default_mode="warn" function instancenorm(args...)
    return _instancenorm(args...)
end

# Needed for Zygote type-stability
function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(instancenorm), args...)
    return CRC.rrule_via_ad(cfg, _instancenorm, args...)
end

function _instancenorm(x::AbstractArray{<:Real, N}, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, training::Val,
        σ::F=identity, epsilon::Real=1.0f-5) where {N, F}
    _test_valid_instancenorm_arguments(x)

    x_, xm, xv = _normalization(x, nothing, nothing, scale, bias,
        _get_instancenorm_reduce_dims(x), training, nothing, epsilon, σ)

    return x_, (; running_mean=xm, running_var=xv)
end

@generated function _get_instancenorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return :($(Val(Tuple([1:(N - 2)]...))))
end

function _test_valid_instancenorm_arguments(::AbstractArray{T, N}) where {T, N}
    N > 2 || throw(ArgumentError("`ndims(x) = $(N)` must be at least 2."))
    return nothing
end

CRC.@non_differentiable _test_valid_instancenorm_arguments(::Any...)
EnzymeRules.inactive_noinl(::typeof(_test_valid_instancenorm_arguments), ::Any...) = nothing

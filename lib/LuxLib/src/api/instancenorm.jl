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
function instancenorm(x::AbstractArray{T, N}, scale::Optional{<:AbstractArray{T, N}},
        bias::Optional{<:AbstractArray{T, N}}, σ::F=identity,
        epsilon::Real=Utils.default_epsilon(x),
        training::Union{Val, StaticBool}=Val(false)) where {T, N, F}
    assert_valid_instancenorm_arguments(x)

    y, xμ, xσ² = Impl.normalization(
        x, nothing, nothing, scale, bias, static(training), nothing,
        epsilon, Impl.select_fastest_activation(σ, x, scale, bias))

    return y, (; running_mean=xμ, running_var=xσ²)
end

function assert_valid_instancenorm_arguments(::AbstractArray{T, N}) where {T, N}
    @assert N>2 "`ndims(x) = $(N)` must be at least > 2."
    return nothing
end

CRC.@non_differentiable assert_valid_instancenorm_arguments(::Any...)

@doc doc"""
    groupnorm(x, scale, bias, groups, σ::F=identity,
        epsilon::Real=eps(eltype(x)) ^ (5 // 7))

Group Normalization. For details see [1].

This op is similar to batch normalization, but statistics are shared across equally-sized
groups of channels and not shared across batch dimension. Thus, group normalization does not
depend on the batch composition and does not require maintaining internal state for storing
statistics.

## Arguments

  - `x`: Input to be Normalized
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)
  - `groups`: Number of groups
  - `σ`: Activation function (default: `identity`)
  - `epsilon`: Value added to the denominator for numerical stability
    (default: `eps(eltype(x)) ^ (5 / 7)`)

## Returns

The normalized array is returned.

## References

[1] Wu, Yuxin, and Kaiming He. "Group normalization." Proceedings of the European conference
    on computer vision (ECCV). 2018.
"""
function groupnorm(x::AbstractArray{<:Real, N}, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, groups::Int, σ::F=identity,
        epsilon::Real=__default_epsilon(x)) where {F, N}
    _test_valid_groupnorm_arguments(x, scale, bias, groups)

    sz = size(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_ = _groupnorm_impl(x_reshaped, scale, bias, _get_groupnorm_reduce_dims(x), epsilon,
        select_fastest_activation(σ, x, scale, bias, x_reshaped))

    return reshape(x_, sz)
end

@generated function _get_groupnorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return :($(Val(Tuple(collect(1:(N - 1))))))
end

function _test_valid_groupnorm_arguments(
        x::AbstractArray{T, N}, scale, bias, groups) where {T, N}
    if scale !== nothing && bias !== nothing && length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of \
                             channels (N - 1 dim of the input array)."))
    end
    if size(x, N - 1) % groups != 0
        throw(ArgumentError("Number of channels $(size(x, N - 1)) must be divisible by \
                             the number of groups $groups."))
    end
    return nothing
end

CRC.@non_differentiable _test_valid_groupnorm_arguments(::Any...)
EnzymeRules.inactive_noinl(::typeof(_test_valid_groupnorm_arguments), ::Any...) = nothing

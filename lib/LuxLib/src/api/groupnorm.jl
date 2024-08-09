@doc doc"""
    groupnorm(x, scale, bias, groups::Int, σ::F=identity,
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
        epsilon::Real=Utils.default_epsilon(x)) where {F, N}
    assert_valid_groupnorm_arguments(x, scale, bias, groups)

    return Impl.groupnorm(x, scale, bias, groups, σ, epsilon)
end

function assert_valid_groupnorm_arguments(
        x::AbstractArray{T, N}, scale, bias, groups) where {T, N}
    @assert length(scale)==length(bias)==size(x, N - 1) "Length of `scale` and `bias` must \
                                                         be equal to the number of \
                                                         channels ((N - 1) dim of the \
                                                         input array)."
    assert_valid_groupnorm_arguments(x, nothing, nothing, groups)
    return nothing
end

function assert_valid_groupnorm_arguments(
        x::AbstractArray{T, N}, ::Nothing, ::Nothing, groups::Int) where {T, N}
    @assert size(x, N - 1) % groups==0 "Number of channels $(size(x, N - 1)) must be \
                                        divisible by the number of groups $groups."
    return nothing
end

CRC.@non_differentiable assert_valid_groupnorm_arguments(::Any...)

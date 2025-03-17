@doc doc"""
    groupnorm(x, scale, bias, groups::Int, σ::F=identity,
        epsilon=eps(eltype(x)) ^ (5 // 7))

Group Normalization. For details see [wu2018group](@citet).

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
"""
function groupnorm(
    x::AbstractArray{<:Number,N},
    scale::Optional{<:AbstractVector},
    bias::Optional{<:AbstractVector},
    groups::Int,
    σ::F=identity,
    epsilon=default_epsilon(x),
) where {F,N}
    assert_valid_groupnorm_arguments(x, scale, bias, groups)
    return groupnorm_impl(
        x, scale, bias, groups, select_fastest_activation(σ, x, scale, bias), epsilon
    )
end

function assert_valid_groupnorm_arguments(
    x::AbstractArray{T,N}, scale, bias, groups
) where {T,N}
    @assert length(scale) == length(bias) == size(x, N - 1) "Length of `scale` and `bias` must \
                                                         be equal to the number of \
                                                         channels ((N - 1) dim of the \
                                                         input array)."
    assert_valid_groupnorm_arguments(x, nothing, nothing, groups)
    return nothing
end

function assert_valid_groupnorm_arguments(
    x::AbstractArray{T,N}, ::Nothing, ::Nothing, groups::Int
) where {T,N}
    @assert size(x, N - 1) % groups == 0 "Number of channels $(size(x, N - 1)) must be \
                                        divisible by the number of groups $groups."
    return nothing
end

CRC.@non_differentiable assert_valid_groupnorm_arguments(::Any...)

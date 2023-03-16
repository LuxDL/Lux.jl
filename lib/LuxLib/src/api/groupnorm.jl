@doc doc"""
    groupnorm(x, scale, bias; groups, epsilon)

Group Normalization. For details see [1].

This op is similar to batch normalization, but statistics are shared across equally-sized
groups of channels and not shared across batch dimension. Thus, group normalization does not
depend on the batch composition and does not require maintaining internal state for storing
statistics.

## Arguments

  - `x`: Input to be Normalized
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)

## Keyword Arguments

  - `groups`: Number of groups
  - `epsilon`: Value added to the denominator for numerical stability

## Returns

Normalized array is returned.

## Performance Considerations

The most common case of this Op -- `x` is a 4D array and there is no statistics tracking --
is optimized using KernelAbstractions and has a fast custom backwards pass implemented. All
other cases have a fallback implementation which is not especially optimized.

Additionally, if the element types of `x`, `scale`, and `bias` are not same and not one of
`Float32` and `Float64`, then the Op uses the slower fallback implementation. We have tested
the code path for `Float16` and it works, but gradient accumulation is extremely fragile.
Hence, for `Float16` inputs, it uses the fallback implementation.

If the batch size is small (< 16), then the fallback implementation will be faster than the
KA version. However, this customization is not possible using the direct `groupnorm`
interface.

## References

[1] Wu, Yuxin, and Kaiming He. "Group normalization." Proceedings of the European conference
    on computer vision (ECCV). 2018.
"""
function groupnorm(x::AbstractArray{T, 4}, scale::AbstractVector{T},
                   bias::AbstractVector{T}; groups::Int,
                   epsilon::Real) where {T <: _GROUPNORM_IMPL_FLOAT}
    _assert_same_device(x, scale, bias)
    if length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("""Length of `scale` and `bias` must be equal to the number of
                               channels (N - 1 dim of the input array)."""))
    end
    if size(x, 3) % groups != 0
        throw(ArgumentError("""Number of channels $(size(x, 3)) must be divisible by the
                               number of groups $groups."""))
    end

    return first(_groupnorm(x, groups, scale, bias, T(epsilon)))
end

# For any reason if the fast path is not possible, then we use the fallback implementation
function groupnorm(x::AbstractArray{<:Real, N},
                   scale::Union{Nothing, AbstractVector{<:Real}},
                   bias::Union{Nothing, AbstractVector{<:Real}}; groups::Int,
                   epsilon::Real) where {N}
    _assert_same_device(x, scale, bias)
    if scale !== nothing && bias !== nothing && length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("""Length of `scale` and `bias` must be equal to the number of
                               channels (N - 1 dim of the input array)."""))
    end
    if size(x, N - 1) % groups != 0
        throw(ArgumentError("""Number of channels $(size(x, 3)) must be divisible by the
                               number of groups $groups."""))
    end

    sz = size(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_, xmean, xvar = _normalization(x_reshaped, nothing, nothing, scale, bias,
                                     _get_groupnorm_reduce_dims(x), Val(true),
                                     zero(eltype(x)), epsilon)

    return reshape(x_, sz)
end

@generated function _get_groupnorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return :($(Val(Tuple(collect(1:(N - 1))))))
end

# Custom Pullbacks
function CRC.rrule(::typeof(groupnorm), x::AbstractArray{T, 4}, scale::AbstractVector{T},
                   bias::AbstractVector{T}; groups::Int,
                   epsilon::Real) where {T <: _GROUPNORM_IMPL_FLOAT}
    _assert_same_device(x, scale, bias)
    if scale !== nothing && bias !== nothing && length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("""Length of `scale` and `bias` must be equal to the number of
                               channels (N - 1 dim of the input array)."""))
    end
    if size(x, 3) % groups != 0
        throw(ArgumentError("""Number of channels $(size(x, 3)) must be divisible by the
                               number of groups $groups."""))
    end

    y, mu, rsig = _groupnorm(x, groups, scale, bias, epsilon)
    function groupnorm_pullback(dy)
        dx, dscale, dbias = _dgroupnorm(dy, y, x, groups, scale, bias, mu, rsig)
        return NoTangent(), dx, dscale, dbias
    end
    return y, groupnorm_pullback
end

@doc doc"""
    groupnorm(x, scale, bias; groups, epsilon)
    groupnorm(x, scale, bias, running_mean, running_var; groups, momentum, training,
              epsilon)

Group Normalization. For details see [1].

This op is similar to batch normalization, but statistics are shared across equally-sized
groups of channels and not shared across batch dimension. Thus, group normalization does not
depend on the batch composition and does not require maintaining internal state for storing
statistics.

## Arguments

  - `x`: Input to be Normalized
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)
  - `running_mean`: Running mean of the inputs. Must be an `AV` or `nothing`.
  - `running_var`: Running variance of the inputs. Must be an `AV` or `nothing`.

## Keyword Arguments

  - `groups`: Number of groups
  - `momentum`: Momentum for updating running mean and variance.
  - `training`: Set to `Val(true)` if running in training mode.
  - `epsilon`: Value added to the denominator for numerical stability

## Returns

If using the first function signature, then the only the normalized array is returned.

Otherwise, the normalized array and a named tuple containing updated running mean and
updated running variance are returned.

## Additional Notes

`running_mean`, `running_var`, `momentum`, and `training` exist only for backwards
compatibility reasons. There is no well documented evidence in literature that tracking
statistics for group normalization actually helps. It is recommended to not use these
arguments at all.

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
function groupnorm(x::AA{T, 4},
    scale::AV{T},
    bias::AV{T};
    groups::Int,
    epsilon::Real) where {T <: FP_32_64}
    _assert_same_backend(x, scale, bias)
    if length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of channels (N - 1 dim of the input array)."))
    end
    if size(x, 3) % groups != 0
        throw(ArgumentError("Number of channels $(size(x, 3)) must be divisible by the number of groups $groups."))
    end

    return first(_groupnorm(x, groups, scale, bias, T(epsilon)))
end

function groupnorm(x::AA{T, 4},
    scale::AV{T},
    bias::AV{T},
    ::Nothing,
    ::Nothing;
    groups::Int,
    epsilon::Real,
    momentum=0.9f0,
    training::Val=Val(true)) where {T <: FP_32_64}
    return groupnorm(x, scale, bias; groups, epsilon),
    (running_mean=nothing, running_var=nothing)
end

# For any reason if the fast path is not possible, then we use the fallback implementation
function groupnorm(x::AA, scale::AV, bias::AV; groups::Int, epsilon::Real)
    return groupnorm(x,
        scale,
        bias,
        nothing,
        nothing;
        groups,
        epsilon,
        momentum=eltype(x)(0.9),
        training=Val(true))[1]
end

# Slow Fallback (without custom Pullback Implementation)
function groupnorm(x::AA{<:Real, N},
    scale::NOrAVR,
    bias::NOrAVR,
    running_mean::NOrAVR,
    running_var::NOrAVR;
    groups::Int,
    momentum::Real,
    training::Val,
    epsilon::Real) where {N}
    _assert_same_backend(x, scale, bias, running_mean, running_var)
    if scale !== nothing && bias !== nothing && length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of channels (N - 1 dim of the input array)."))
    end
    if size(x, N - 1) % groups != 0
        throw(ArgumentError("Number of channels $(size(x, 3)) must be divisible by the number of groups $groups."))
    end

    sz = size(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_, xmean, xvar = _normalization(x_reshaped,
        running_mean,
        running_var,
        scale,
        bias,
        _get_groupnorm_reduce_dims(x),
        training,
        momentum,
        epsilon)

    return reshape(x_, sz), (; running_mean=xmean, running_var=xvar)
end

@generated function _get_groupnorm_reduce_dims(::AA{T, N}) where {T, N}
    return :($(Val(Tuple(collect(1:(N - 1))))))
end

# Custom Pullbacks
function CRC.rrule(::typeof(groupnorm),
    x::AA{T, 4},
    scale::AV{T},
    bias::AV{T};
    groups::Int,
    epsilon::Real) where {T <: FP_32_64}
    _assert_same_backend(x, scale, bias)
    if length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of channels (N - 1 dim of the input array)."))
    end
    if size(x, 3) % groups != 0
        throw(ArgumentError("Number of channels $(size(x, 3)) must be divisible by the number of groups $groups."))
    end

    y, mu, rsig = _groupnorm(x, groups, scale, bias, epsilon)
    function groupnorm_pullback(dy)
        dx, dscale, dbias = _dgroupnorm(dy, y, x, groups, scale, bias, mu, rsig)
        return ∂∅, dx, dscale, dbias
    end
    return y, groupnorm_pullback
end

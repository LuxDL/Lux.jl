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

The normalized array is returned.

## Performance Considerations

The most common case of this Op -- `x` is a 4D array -- is optimized using
KernelAbstractions and has a fast custom backwards pass implemented. All other cases have a
fallback implementation which is not especially optimized.

We have tested the code path for `Float16` and it works, but gradient accumulation is
extremely fragile. Hence, for `Float16` inputs, it uses the fallback implementation.

If the batch size is small (< 16), then the fallback implementation will be faster than the
KA version. However, this customization is not possible using the direct `groupnorm`
interface.

## References

[1] Wu, Yuxin, and Kaiming He. "Group normalization." Proceedings of the European conference
    on computer vision (ECCV). 2018.
"""
function groupnorm(x::AbstractArray{<:Union{Float32, Float64}, 4},
        scale::AbstractVector{<:Union{Float32, Float64}},
        bias::AbstractVector{<:Union{Float32, Float64}},
        σ::F=identity; groups::Int, epsilon::Real) where {F}
    _assert_same_backend(x, scale, bias)
    if length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of \
                             channels (N - 1 dim of the input array)."))
    end
    if size(x, 3) % groups != 0
        throw(ArgumentError(lazy"Number of channels $(size(x, 3)) must be divisible by the number of groups $groups."))
    end

    # FIXME: We need to fuse the activation function into the kernel for optimal performance
    return fast_activation!!(σ, __fast_groupnorm(x, groups, scale, bias, epsilon))
    # return σ.(__fast_groupnorm(x, groups, scale, bias, epsilon))
end

# Separate this out for a cleaner rrule later on
@inline function __fast_groupnorm(x, groups, scale, bias, epsilon)
    return first(_groupnorm(x, groups, scale, bias, epsilon))
end

# Slow Fallback (without custom Pullback Implementation)
function groupnorm(x::AbstractArray{<:Real, N}, scale::Union{Nothing, <:AbstractVector},
        bias::Union{Nothing, <:AbstractVector}, σ::F=identity;
        groups::Int, epsilon::Real) where {F, N}
    _assert_same_backend(x, scale, bias)
    if scale !== nothing && bias !== nothing && length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of \
                             channels (N - 1 dim of the input array)."))
    end
    if size(x, N - 1) % groups != 0
        throw(ArgumentError(lazy"Number of channels $(size(x, 3)) must be divisible by the number of groups $groups."))
    end

    sz = size(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_ = first(_normalization(x_reshaped, nothing, nothing, scale, bias,
        _get_groupnorm_reduce_dims(x), Val(false), nothing, epsilon, σ))

    return reshape(x_, sz)
end

@generated function _get_groupnorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return :($(Val(Tuple(collect(1:(N - 1))))))
end

# Custom Pullbacks
function CRC.rrule(::typeof(__fast_groupnorm), x, groups, scale, bias, epsilon)
    y, μ, σ⁻¹ = _groupnorm(x, groups, scale, bias, epsilon)
    ∇groupnorm = @closure Δ -> begin
        ∂x, ∂scale, ∂bias = _∇groupnorm(Δ, y, x, groups, scale, bias, μ, σ⁻¹)
        return CRC.NoTangent(), ∂x, CRC.NoTangent(), ∂scale, ∂bias, CRC.NoTangent()
    end
    return y, ∇groupnorm
end

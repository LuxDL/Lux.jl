@doc doc"""
    groupnorm(x, scale, bias, groups, σ::F=identity, epsilon::Real=1.0f-5)

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
  - `epsilon`: Value added to the denominator for numerical stability (default: `1f-5`)

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
        groups::Int, σ::F=identity, epsilon::Real=1.0f-5) where {F}
    _test_valid_groupnorm_arguments(x, scale, bias, groups)
    # FIXME: We need to fuse the activation function into the kernel for optimal performance
    return fast_activation!!(
        σ, __groupnorm_kernel_abstractions(x, groups, scale, bias, epsilon))
end

# Slow Fallback (without custom Pullback Implementation)
function groupnorm(x::AbstractArray{<:Real, N}, scale::Union{Nothing, <:AbstractVector},
        bias::Union{Nothing, <:AbstractVector}, groups::Int,
        σ::F=identity, epsilon::Real=1.0f-5) where {F, N}
    _test_valid_groupnorm_arguments(x, scale, bias, groups)

    sz = size(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_ = first(_normalization(x_reshaped, nothing, nothing, scale, bias,
        _get_groupnorm_reduce_dims(x), Val(false), nothing, epsilon, σ))

    return reshape(x_, sz)
end

@generated function _get_groupnorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return :($(Val(Tuple(collect(1:(N - 1))))))
end

function _test_valid_groupnorm_arguments(
        x::AbstractArray{T, N}, scale, bias, groups) where {T, N}
    _assert_same_backend(x, scale, bias)
    if scale !== nothing && bias !== nothing && length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of \
                             channels (N - 1 dim of the input array)."))
    end
    if size(x, N - 1) % groups != 0
        throw(ArgumentError(lazy"Number of channels $(size(x, N - 1)) must be divisible by the number of groups $groups."))
    end
    return nothing
end

CRC.@non_differentiable _test_valid_groupnorm_arguments(::Any...)
EnzymeRules.inactive(::typeof(_test_valid_groupnorm_arguments), ::Any...) = nothing

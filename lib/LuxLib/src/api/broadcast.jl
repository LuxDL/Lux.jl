## bypass a type instability
function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(fast_activation!!),
        σ::F, x::AbstractArray{T}) where {F, T}
    return CRC.rrule_via_ad(cfg, fast_broadcast!!, σ, x)
end

"""
    fast_broadcast!!(f::F, x::AbstractArray, args...) where {F}

if `x` is an immutable array, it computes `@. f(x, args...)`. Otherwise, it computes
`@. x = f(x, args...)`.

Additionally, whether `x` is updated in-place, depends on whether this function is being
called inside a differentiated function.
"""
function fast_broadcast!!(f::F, x::AbstractArray, args...) where {F <: Function}
    return _fast_broadcast!!(Val(ArrayInterface.can_setindex(typeof(x))), f, x, args...)
end

# Generic fallback. We define specialized fallbacks in the impl file
function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(fast_broadcast!!),
        f::F, x::AbstractArray, args...) where {F}
    return CRC.rrule_via_ad(cfg, broadcast, f, x, args...)
end

function _fast_broadcast!!(
        ::Val{true}, f::F, x::AbstractArray, args...) where {F <: Function}
    return _fast_broadcast!(f, x, args...)
end
function _fast_broadcast!!(
        ::Val{false}, f::F, x::AbstractArray, args...) where {F <: Function}
    return _fast_broadcast(f, x, args...)
end

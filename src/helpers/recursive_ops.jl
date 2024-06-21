"""
    recursive_add!!(x, y)

Recursively add the leaves of two nested structures `x` and `y`. In Functor language, this
is equivalent to doing `fmap(+, x, y)`, but this implementation uses type stable code for
common cases.

Any leaves of `x` that are arrays and allow in-place addition will be modified in place.
"""
function recursive_add!!(x::AbstractArray, y::AbstractArray)
    ArrayInterface.can_setindex(x) || return x .+ y
    @. x += y
    return x
end
recursive_add!!(x::Tuple, y::Tuple) = map(recursive_add!!, x, y)
recursive_add!!(::Nothing, ::Nothing) = nothing
function recursive_add!!(x::NamedTuple{F}, y::NamedTuple{F}) where {F}
    return NamedTuple{F}(map(recursive_add!!, values(x), values(y)))
end
recursive_add!!(x, y) = fmap(recursive_add!!, x, y)

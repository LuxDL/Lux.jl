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

"""
    recursive_eltype(x)

Recursively determine the element type of a nested structure `x`. This is equivalent to
doing `fmap(eltype, x)`, but this implementation uses type stable code for common cases.
"""
@inline recursive_eltype(x::AbstractArray) = eltype(x)
@inline recursive_eltype(x::Tuple) = promote_type(__recursice_eltype.(x)...)
@inline recursive_eltype(x::NamedTuple) = promote_type(recursive_eltype.(values(x))...)
@inline recursive_eltype(::Nothing) = Bool
@inline recursive_eltype(x::Number) = eltype(x)
@inline function recursive_eltype(x)
    _eltype = Ref(Bool)
    function __internal_recursive_eltype(x)
        _eltype[] = promote_type(_eltype[], recursive_eltype(x))
        return x
    end
    fmap(__internal_recursive_eltype, x)
    return _eltype[]
end

"""
    recursive_make_zero(x)

Recursively create a zero value for a nested structure `x`. This is equivalent to doing
`fmap(zero, x)`, but this implementation uses type stable code for common cases.

See also [`Lux.recursive_make_zero!!`](@ref).
"""
@inline recursive_make_zero(x::Number) = zero(x)
@inline recursive_make_zero(x::AbstractArray{<:Number}) = zero(x)
@inline recursive_make_zero(x::AbstractArray) = map(recursive_make_zero, x)
@inline recursive_make_zero(x::Tuple) = map(recursive_make_zero, x)
@inline recursive_make_zero(x::NamedTuple{fields}) where {fields} = NamedTuple{fields}(map(
    recursive_make_zero, values(x)))
@inline recursive_make_zero(::Nothing) = nothing
@inline recursive_make_zero(v::Val) = v
@inline recursive_make_zero(x) = fmap(recursive_make_zero, x)

"""
    recursive_make_zero!!(x)

Recursively create a zero value for a nested structure `x`. Leaves that can be mutated with
in-place zeroing will be modified in place.

See also [`Lux.recursive_make_zero`](@ref) for fully out-of-place version.
"""
@inline recursive_make_zero!!(x::Number) = zero(x)
@inline recursive_make_zero!!(x::AbstractArray{<:Number}) = fill!(x, zero(eltype(x)))
@inline recursive_make_zero!!(x::AbstractArray) = map(recursive_make_zero!!, x)
@inline recursive_make_zero!!(x::Tuple) = map(recursive_make_zero!!, x)
@inline recursive_make_zero!!(x::NamedTuple{fields}) where {fields} = NamedTuple{fields}(map(
    recursive_make_zero!!, values(x)))
@inline recursive_make_zero!!(::Nothing) = nothing
@inline recursive_make_zero!!(x) = fmap(recursive_make_zero!!, x)

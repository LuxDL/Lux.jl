"""
    recursive_add!!(x, y)

Recursively add the leaves of two nested structures `x` and `y`. In Functor language, this
is equivalent to doing `fmap(+, x, y)`, but this implementation uses type stable code for
common cases.

Any leaves of `x` that are arrays and allow in-place addition will be modified in place.
"""
recursive_add!!(x, y) = recursive_map(Utils.add!!, x, y)

"""
    recursive_eltype(x, unwrap_ad_types = Val(false))

Recursively determine the element type of a nested structure `x`. This is equivalent to
doing `fmap(Lux.Utils.eltype, x)`, but this implementation uses type stable code for common
cases.

For ambiguous inputs like `nothing` and `Val` types we return `Bool` as the eltype.

If `unwrap_ad_types` is set to `Val(true)` then for tracing and operator overloading based
ADs (ForwardDiff, ReverseDiff, Tracker), this function will return the eltype of the
unwrapped value.
"""
recursive_eltype(x) = recursive_eltype(x, Val(false))
recursive_eltype(x, val::Val) = recursive_eltype(x, static(val))

recursive_eltype(x::AbstractArray, ::False) = eltype(x)
recursive_eltype(x::AbstractArray, ::True) = Utils.eltype(x)
recursive_eltype(x::Number, ::False) = eltype(x)
recursive_eltype(x::Number, ::True) = Utils.eltype(x)
recursive_eltype(::Union{Nothing, Missing, Val}, ::StaticBool) = Bool
function recursive_eltype(x::Union{Tuple, NamedTuple}, val::StaticBool)
    leaves = x isa Tuple ? x : values(x)
    length(leaves) == 0 && return Bool
    return mapreduce(Base.Fix2(recursive_eltype, val), promote_type, leaves)
end
function recursive_eltype(x, val::StaticBool)
    leaves = x isa Tuple ? x : (x isa NamedTuple ? values(x) : Functors.fleaves(x))
    length(leaves) == 0 && return Bool
    return mapreduce(Base.Fix2(recursive_eltype, val), promote_type, leaves)
end

"""
    recursive_make_zero(x)

Recursively create a zero value for a nested structure `x`. This is equivalent to doing
`fmap(zero, x)`, but this implementation uses type stable code for common cases.

See also [`Lux.recursive_make_zero!!`](@ref).
"""
recursive_make_zero(x) = recursive_map(Utils.zero, x)

"""
    recursive_make_zero!!(x)

Recursively create a zero value for a nested structure `x`. Leaves that can be mutated with
in-place zeroing will be modified in place.

See also [`Lux.recursive_make_zero`](@ref) for fully out-of-place version.
"""
recursive_make_zero!!(x) = recursive_map(Utils.zero!!, x)

"""
    recursive_copyto!(x, y)

Recursively copy the leaves of two nested structures `x` and `y`. In Functor language, this
is equivalent to doing `fmap(copyto!, x, y)`, but this implementation uses type stable code
for common cases. Note that any immutable leaf will lead to an error.
"""
recursive_copyto!(x, y) = recursive_map(copyto!, x, y)

"""
    recursive_map(f, x, args...)

Similar to `fmap(f, args...)` but with restricted support for the notion of "leaf" types.
However, this allows for more efficient and type stable implementations of recursive
operations.

## How this works?

For the following types it directly defines recursion rules:

 1. `AbstractArray`: If eltype is `isbitstype`, then `f` is applied to the array, else we
    recurse on the array.
 2. `Tuple/NamedTuple`: We recurse on the values.
 3. `Number/Val/Nothing`: We directly apply `f`.
 4. For all other types, we recurse on the fields using `Functors.fmap`.

!!! note

    In most cases, users should gravitate towards `Functors.fmap` if it is being used
    outside of hot loops. Even for other cases, it is always recommended to verify the
    correctness of this implementation for specific usecases.
"""
function recursive_map end

for direct_call in (Number, Val, Nothing)
    @eval recursive_map(f::F, x::$(direct_call), args...) where {F} = f(x, args...)
end
function recursive_map(f::F, x::AbstractArray{T}, args...) where {F, T}
    (T <: Number || isbitstype(T)) && return f(x, args...) # Not all Number types (BigFloat) are bitstype
    return f.(x, args...)
end
function recursive_map(f::F, x::Union{NamedTuple, Tuple}, args...) where {F}
    map_fn = let f = f
        (args_...) -> recursive_map(f, args_...)
    end
    return map(map_fn, x, args...)
end
recursive_map(f::F, x, args...) where {F} = fmap(f, x, args...)

@compat(public,
    (recursive_add!!, recursive_copyto!, recursive_eltype,
    recursive_make_zero, recursive_map, recursive_make_zero!!))

const RECURSIVE_OPS_DEPRECATION_MSG = """
!!! warning "Deprecation Warning"

    Starting Lux v1.3.0, this function is deprecated in favor of `Functors.fmap`. Functors
    v0.5 made significant strides towards improving the performance of `fmap` and hence
    this function has been deprecated. Users are encouraged to use `Functors.fmap` instead.
"""

"""
    recursive_add!!(x, y)

Recursively add the leaves of two nested structures `x` and `y`. In Functor language, this
is equivalent to doing `fmap(+, x, y)`, but this implementation uses type stable code for
common cases.

Any leaves of `x` that are arrays and allow in-place addition will be modified in place.

$(RECURSIVE_OPS_DEPRECATION_MSG)
"""
function recursive_add!! end

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
recursive_eltype(::CRC.ZeroTangent, ::StaticBool) = Bool
recursive_eltype(::CRC.NoTangent, ::StaticBool) = Bool
recursive_eltype(::Union{Nothing,Missing,Val}, ::StaticBool) = Bool
function recursive_eltype(x::Union{Tuple,NamedTuple}, val::StaticBool)
    leaves = x isa Tuple ? x : values(x)
    return mapreduce(Base.Fix2(recursive_eltype, val), promote_type, leaves; init=Bool)
end
function recursive_eltype(x, val::StaticBool)
    leaves = Functors.fleaves(x; exclude=MLDataDevices.isleaf)
    return mapreduce(Base.Fix2(recursive_eltype, val), promote_type, leaves; init=Bool)
end

"""
    recursive_make_zero(x)

Recursively create a zero value for a nested structure `x`. This is equivalent to doing
`fmap(zero, x)`, but this implementation uses type stable code for common cases.

See also [`Lux.recursive_make_zero!!`](@ref).

$(RECURSIVE_OPS_DEPRECATION_MSG)
"""
function recursive_make_zero end

"""
    recursive_make_zero!!(x)

Recursively create a zero value for a nested structure `x`. Leaves that can be mutated with
in-place zeroing will be modified in place.

See also [`Lux.recursive_make_zero`](@ref) for fully out-of-place version.

$(RECURSIVE_OPS_DEPRECATION_MSG)
"""
function recursive_make_zero!! end

"""
    recursive_copyto!(x, y)

Recursively copy the leaves of two nested structures `x` and `y`. In Functor language, this
is equivalent to doing `fmap(copyto!, x, y)`, but this implementation uses type stable code
for common cases. Note that any immutable leaf will lead to an error.

$(RECURSIVE_OPS_DEPRECATION_MSG)
"""
function recursive_copyto! end

"""
    recursive_map(f, x, args...)

Similar to `fmap(f, args...)` but with restricted support for the notion of "leaf" types.
However, this allows for more efficient and type stable implementations of recursive
operations.

$(RECURSIVE_OPS_DEPRECATION_MSG)

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

@public recursive_eltype

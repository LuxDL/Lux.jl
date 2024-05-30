@inline function __make_concrete_array(x)
    return Reactant.make_tracer(IdDict(), x, (), Reactant.ArrayToConcrete, nothing)
end

@inline function __try_similar_structure(x::AbstractArray, y::NamedTuple{()})
    length(x) == 0 && return y
    throw(DimensionMismatch(lazy"Expected empty array, got $(size(x))."))
end
@inline function __try_similar_structure(x::AbstractArray, y::AbstractArray)
    return parent(x) !== x ? copy(x) : x # unview arrays and such
end
@inline __try_similar_structure(x, y) = fmap(__try_similar_structure, x, y)

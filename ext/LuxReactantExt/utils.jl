@inline Lux.__make_reactant_array(x::Reactant.RArray) = x
@inline function Lux.__make_reactant_array(x::AbstractArray)
    hasmethod(Reactant.ArrayToConcrete, Tuple{typeof(x)}) &&
        return Reactant.ConcreteRArray(x)
    return __make_tracer(x)
end
@inline Lux.__make_reactant_array(x) = __make_tracer(x)

@inline function __make_tracer(x)
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

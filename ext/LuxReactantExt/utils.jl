@inline Lux.__make_reactant_array(x::Reactant.RArray) = x
@inline function Lux.__make_reactant_array(x::AbstractArray)
    hasmethod(Reactant.ArrayToConcrete, Tuple{typeof(x)}) &&
        return Reactant.ConcreteRArray(x)
    return __make_tracer(x)
end
@inline Lux.__make_reactant_array(x) = __make_tracer(x)

@inline function __make_tracer(x)
    return Reactant.make_tracer(IdDict(), x, (), Reactant.ArrayToConcrete)
end

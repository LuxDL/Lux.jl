__make_reactant_array(x::Reactant.RArray) = x
function __make_reactant_array(x::AbstractArray)
    hasmethod(Reactant.ArrayToConcrete, Tuple{typeof(x)}) &&
        return Reactant.ConcreteRArray(x)
    return __make_tracer(x)
end
__make_reactant_array(x) = __make_tracer(x)

__make_tracer(x) = Reactant.make_tracer(IdDict(), x, (), Reactant.ArrayToConcrete)

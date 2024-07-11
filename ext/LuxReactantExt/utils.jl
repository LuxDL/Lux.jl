__make_reactant_array(x::Reactant.RArray) = x
function __make_reactant_array(x::AbstractArray)
    hasmethod(Reactant.ArrayToConcrete, Tuple{typeof(x)}) &&
        return Reactant.ConcreteRArray(x)
    return __make_tracer(x)
end
function __make_reactant_array(x)
    return Lux.recursive_map(x) do xₗ
        hasmethod(Reactant.ArrayToConcrete, Tuple{typeof(xₗ)}) &&
            return Reactant.ConcreteRArray(xₗ)
        return __make_tracer(xₗ)
    end
end

__make_tracer(x) = Reactant.make_tracer(IdDict(), x, (), Reactant.ArrayToConcrete)

module LuxReactantExt

using ArgCheck: @argcheck
using Enzyme: Enzyme
using Random: AbstractRNG, Xoshiro
using Reactant: Reactant
using Lux: Lux
using LuxCore: LuxCore, AbstractExplicitLayer

@inline __make_concrete_array(x::Reactant.ConcreteRArray) = x
@inline __make_concrete_array(x::AbstractArray) = Reactant.ConcreteRArray(x)
@inline function __make_concrete_array(x)
    return Reactant.make_tracer(IdDict(), x, (), Reactant.ArrayToConcrete, nothing)
end

function Lux.__to_reactant_adaptor(model::AbstractExplicitLayer, input_prototype)
    concrete_input = __make_concrete_array(input_prototype)
    cmodel = __make_concrete_array(model)

    # We generate fake parameters and states to compile the model
    ps = LuxCore.initialparameters(Xoshiro(123), model)
    cps = __make_concrete_array(ps)

    st = LuxCore.initialstates(Xoshiro(123), model)
    cst = __make_concrete_array(st)

    csmodel = Lux.StatefulLuxLayer{false}(cmodel, cps, cst)

    fwd = Reactant.compile((m, x) -> m(x), (csmodel, concrete_input))

    return Lux.ReactantLayer(model, cmodel, fwd, nothing)
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::Lux.ReactantLayer)
    return __make_concrete_array(LuxCore.initialparameters(rng, layer.layer))
end

function LuxCore.initialstates(rng::AbstractRNG, layer::Lux.ReactantLayer)
    return __make_concrete_array(LuxCore.initialstates(rng, layer.layer))
end

function (l::Lux.ReactantLayer)(x, ps, st::NamedTuple)
    csmodel = Lux.StatefulLuxLayer{false}(l.clayer, ps, st)
    y = l.fwd(csmodel, __make_concrete_array(x))
    return y, csmodel.st_any
end

end

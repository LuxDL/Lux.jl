module LuxReactantExt

using ArgCheck: @argcheck
using Random: AbstractRNG, Xoshiro
using Reactant: Reactant
using Lux: Lux
using LuxCore: LuxCore, AbstractExplicitLayer

@inline __make_concrete_array(x::Reactant.ConcreteRArray) = x
@inline __make_concrete_array(x::AbstractArray) = Reactant.ConcreteRArray(x)
@inline function __make_concrete_array(x)
    return Reactant.make_tracer(IdDict(), x, (), Reactant.ArrayToConcrete, nothing)
end

# FIXME: currently only `stateless_apply` is supported: https://github.com/EnzymeAD/Reactant.jl/issues/8
function Lux.__to_reactant_adaptor(model::AbstractExplicitLayer, input_prototype)
    concrete_input = __make_concrete_array(input_prototype)
    cmodel = __make_concrete_array(model)
    # We generate fake parameters and states to compile the model
    ps = LuxCore.initialparameters(Xoshiro(123), model)
    cps = __make_concrete_array(ps)

    st = LuxCore.initialstates(Xoshiro(123), model)
    @argcheck st==LuxCore._getemptystate(model) "Currently only stateless models are supported."

    fwd = Reactant.compile(
        (m, x, ps) -> LuxCore.stateless_apply(m, x, ps), (cmodel, concrete_input, cps))

    # TODO: conditionally compile the backward pass

    return Lux.ReactantLayer(model, cmodel, fwd, nothing)
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::Lux.ReactantLayer)
    return __make_concrete_array(LuxCore.initialparameters(rng, layer.layer))
end

# FIXME: Change once https://github.com/EnzymeAD/Reactant.jl/pull/8 is fixed
function LuxCore.initialstates(::AbstractRNG, layer::Lux.ReactantLayer)
    return NamedTuple() # __make_concrete_array(LuxCore.initialstates(rng, layer.layer))
end

# TODO: Add a type assert here to make it type stable
function (l::Lux.ReactantLayer)(x, ps, ::NamedTuple{()})
    return LuxCore.stateless_apply(l.clayer, __make_concrete_array(x), ps), NamedTuple()
end

end

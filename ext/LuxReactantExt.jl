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

function Lux.__to_reactant_adaptor(
        to::Lux.ToReactantAdaptor{FST}, model::AbstractExplicitLayer) where {FST}
    concrete_input = __make_concrete_array(to.input_prototype)
    cmodel = __make_concrete_array(model)

    # We generate fake parameters and states to compile the model
    ps = LuxCore.initialparameters(Xoshiro(123), model)
    cps = __make_concrete_array(ps)

    st = LuxCore.initialstates(Xoshiro(123), model)
    cst = __make_concrete_array(st)

    csmodel = Lux.StatefulLuxLayer{FST}(cmodel, cps, cst)

    fwd = Reactant.compile((m, x) -> m(x), (csmodel, concrete_input))

    bwd = try
        enzyme_grad_fn = (m, x) -> begin
            dx = Enzyme.make_zero(x)
            dps = Enzyme.make_zero(m.ps)
            st = ifelse(FST, m.st, m.st_any)
            Enzyme.autodiff(
                Enzyme.Reverse, (m, x, ps, st) -> first(LuxCore.apply(m, x, ps, st)),
                Enzyme.Duplicated, Enzyme.Const(m), Enzyme.Duplicated(x, dx),
                Enzyme.Duplicated(ps, dps), Enzyme.Const(st))
            return (; ps=dps), dx
        end

        Reactant.compile(enzyme_grad_fn, (csmodel, concrete_input))
    catch err
        to.force_compile_backward && rethrow(err)
        @error "Enzyme failed to compile the backward pass. Differentiation will be \
                disabled for this model." exception=err
        nothing
    end

    return Lux.ReactantLayer{FST}(model, cmodel, fwd, bwd)
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::Lux.ReactantLayer)
    return __make_concrete_array(LuxCore.initialparameters(rng, layer.layer))
end

function LuxCore.initialstates(rng::AbstractRNG, layer::Lux.ReactantLayer)
    return __make_concrete_array(LuxCore.initialstates(rng, layer.layer))
end

function (l::Lux.ReactantLayer{FST})(x, ps, st::NamedTuple) where {FST}
    csmodel = Lux.StatefulLuxLayer{FST}(l.clayer, ps, st)
    y = l.fwd(csmodel, __make_concrete_array(x))
    return y, ifelse(FST, csmodel.st, csmodel.st_any)
end

end

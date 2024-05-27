module LuxReactantExt

using Adapt: adapt
using ArgCheck: @argcheck
using Enzyme: Enzyme
using Functors: fmapstructure
using Random: AbstractRNG, Xoshiro
using Reactant: Reactant
using Lux: Lux, LuxEltypeAdaptor
using LuxCore: LuxCore, AbstractExplicitLayer

@inline __make_concrete_array(x::Reactant.ConcreteRArray) = x
@inline __make_concrete_array(x::AbstractArray) = Reactant.ConcreteRArray(x)
@inline function __make_concrete_array(x)
    return Reactant.make_tracer(IdDict(), x, (), Reactant.ArrayToConcrete, nothing)
end

# Reactant doesn't handle mixed eltypes that well, so we will first try to compile it as
# a usual julia function. However, if that fails, we will type cast and try to recompile.
# Note that this is only a one time operation so it doesn't matter if this step is too slow.
function Lux.__to_reactant_adaptor(
        to::Lux.ToReactantAdaptor{FST}, model::AbstractExplicitLayer) where {FST}
    input_prototype = to.input_prototype
    input_eltype = Lux.__recursive_eltype(input_prototype)
    ps, st = Lux.setup(Xoshiro(123), model) # We generate fake parameters and states to compile the model
    ps_eltype = Lux.__recursive_eltype(ps)
    st_eltype = Lux.__recursive_eltype(st)

    newT = promote_type(input_eltype, ps_eltype, st_eltype)
    eltype_adaptor = nothing

    if !to.force_allow_mixed_eltypes &&
       any(x -> x != newT && x != Union{}, (input_eltype, ps_eltype, st_eltype))
        # Try compiling, but this might fail
        try
            return Lux.__to_reactant_adaptor(to, model, input_prototype, ps, st, nothing)
        catch err
            @warn """
            Mixed Eltypes detected. Failure is NOT unexpected. Trying to recompile with a \
            common eltype.

            HINT: To force compiling the mixed eltypes, set \
            `force_allow_mixed_eltypes=true` in the constructor of `ToReactantAdaptor`.

            If compilation succeeds, all inputs to the compiled model will be \
            automatically type casted to the common eltype.\n
            """ exception=err input_eltype ps_eltype st_eltype common_eltype=newT
        end

        eltype_adaptor = LuxEltypeAdaptor{newT}()
        input_prototype = adapt(eltype_adaptor, to.input_prototype)
        ps = adapt(eltype_adaptor, ps)
        st = adapt(eltype_adaptor, st)
    end

    return Lux.__to_reactant_adaptor(to, model, input_prototype, ps, st, eltype_adaptor)
end

function Lux.__to_reactant_adaptor(
        to::Lux.ToReactantAdaptor{FST}, model::AbstractExplicitLayer,
        input_prototype, ps, st, eltype_adaptor) where {FST}
    concrete_input = __make_concrete_array(input_prototype)
    cmodel = __make_concrete_array(model)
    cps = __make_concrete_array(ps)
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
        @error """
        Enzyme failed to compile the backward pass. Differentiation will be disabled for \
        this model.

        HINT: To force compilation of the backward pass, set `force_compile_backward=true` \
        in the constructor of `ToReactantAdaptor`.\n
        """ exception=err
        nothing
    end

    return Lux.ReactantLayer{FST, Lux.__recursive_eltype(input_prototype)}(
        model, cmodel, fwd, bwd, eltype_adaptor, fmapstructure(Lux.__size, input_prototype))
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::Lux.ReactantLayer)
    ps = LuxCore.initialparameters(rng, layer.layer)
    layer.eltype_adaptor !== nothing && (ps = adapt(layer.eltype_adaptor, ps))
    return __make_concrete_array(ps)
end

function LuxCore.initialstates(rng::AbstractRNG, layer::Lux.ReactantLayer)
    st = LuxCore.initialstates(rng, layer.layer)
    layer.eltype_adaptor !== nothing && (st = adapt(layer.eltype_adaptor, st))
    return __make_concrete_array(st)
end

function (l::Lux.ReactantLayer{FST, T})(x, ps, st::NamedTuple) where {FST, T}
    csmodel = Lux.StatefulLuxLayer{FST}(l.clayer, ps, st)
    l.eltype_adaptor !== nothing && (x = adapt(l.eltype_adaptor, x))

    # XLARuntimeError is not great, so check and terminate early if needed
    input_structure = fmapstructure(Lux.__size, x)
    if l.input_structure != input_structure
        throw(DimensionMismatch(lazy"Input structure mismatch. Expected $(l.input_structure), got $(input_structure)."))
    end

    # TODO: For non array inputs this we make the eltype uniform which might not be
    # desirable. We should handle those cases with `fmap`
    if T != Lux.__recursive_eltype(x)
        @warn """
        `Reactant.compile` was called with input eltype $(T) but the current input eltype \
        is $(Lux.__recursive_eltype(x)). This might lead to unexpected behavior.

        We will convert the input to $(T) and continue. If you want to avoid this, please \
        recompile the model with the correct input eltype.
        """ maxlog=1
        x = adapt(LuxEltypeAdaptor{T}(), x)
    end

    y = Lux.__apply_reactant(l, csmodel, x)
    return y, ifelse(FST, csmodel.st, csmodel.st_any)
end

@inline Lux.__apply_reactant(l::Lux.ReactantLayer, csmodel, x) = l.fwd(
    csmodel, __make_concrete_array(x))

end

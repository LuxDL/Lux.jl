# Reactant doesn't handle mixed eltypes that well, so we will first try to compile it as
# a usual julia function. However, if that fails, we will type cast and try to recompile.
# Note that this is only a one time operation so it doesn't matter if this step is too slow.
function Lux.__to_reactant_adaptor(
        to::Lux.ToReactantAdaptor{FST}, model::AbstractExplicitLayer) where {FST}
    input_prototype = to.input_prototype
    input_eltype = Lux.__recursive_eltype(input_prototype)
    ps, st = Lux.setup(LuxCore.replicate(to.rng), model)
    ps = to.ps_transform(ps)
    ps_eltype = Lux.__recursive_eltype(ps)
    st_eltype = Lux.__recursive_eltype(st)

    newT = promote_type(input_eltype, ps_eltype, st_eltype)
    eltype_adaptor = nothing

    if !to.force_allow_mixed_eltypes &&
       any(x -> x != newT && x != Union{}, (input_eltype, ps_eltype, st_eltype))
        try # Try compiling, but this might fail
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
    output = first(model(input_prototype, ps, st))
    concrete_output = Lux.__make_reactant_array(output)

    concrete_input = Lux.__make_reactant_array(input_prototype)
    cps = Lux.__make_reactant_array(ps)
    cst = Lux.__make_reactant_array(st)

    smodel = Lux.StatefulLuxLayer{FST}(model, cps, cst)
    fwd_fn = Reactant.compile((m, x) -> m(x), (smodel, concrete_input))

    cst_test = Lux.__make_reactant_array(Lux.testmode(st))
    smodel_test = Lux.StatefulLuxLayer{FST}(model, cps, cst_test)
    inference_fn = Reactant.compile((m, x) -> m(x), (smodel_test, concrete_input))

    vjp_fn = if to.skip_compile_vjp
        nothing
    else
        function enzyme_vjp_fn(m, x, y, dy)
            dx = Enzyme.make_zero(x)
            dps = Enzyme.make_zero(m.ps)
            st_m = ifelse(FST, m.st, m.st_any)

            function wrapper_fn!(y, model, x, ps, st)
                copyto!(y, first(LuxCore.apply(model, x, ps, st)))
                return nothing
            end

            Enzyme.autodiff(
                Enzyme.Reverse, wrapper_fn!, Enzyme.Const, Enzyme.Duplicated(y, dy),
                Enzyme.Const(m.model), Enzyme.Duplicated(x, dx),
                Enzyme.Duplicated(m.ps, dps), Enzyme.Const(st_m))
            return dx, dps
        end

        try
            concrete_output2 = Lux.__make_reactant_array(deepcopy(output))
            Reactant.compile(
                enzyme_vjp_fn, (smodel, concrete_input, concrete_output, concrete_output2))
        catch err
            to.force_compile_backward && rethrow(err)
            @error """
            Enzyme failed to compile the backward pass. Differentiation will be disabled \
            for this model.

            HINT: To force compilation of the backward pass, set \
            `force_compile_backward=true` in the constructor of `ToReactantAdaptor`.\n
            """ exception=err
            nothing
        end
    end

    jvp_fn = if to.skip_compile_jvp
        nothing
    else # TODO: Implement JVP with Enzyme.Forward
        throw(ArgumentError("JVPs are not implemented yet."))
    end

    return Lux.ReactantLayer{
        FST, Lux.__recursive_eltype(input_prototype), typeof(input_prototype),
        typeof(concrete_input), typeof(cst), typeof(cst_test)}(
        to, cps, model, fwd_fn, inference_fn, vjp_fn, jvp_fn,
        eltype_adaptor, fmapstructure(Lux.__size, input_prototype))
end

# TODO: Currently we are maintaining 2 copies of the parameters, this is not ideal.
#       We can return the parameters and states from the layer itself, since we don't care
#       about the values, but just the type.
function LuxCore.initialparameters(rng::AbstractRNG, layer::Lux.ReactantLayer)
    ps = layer.adaptor(LuxCore.initialparameters(rng, layer.layer))
    layer.eltype_adaptor !== nothing && (ps = adapt(layer.eltype_adaptor, ps))
    return Lux.__make_reactant_array(ps)
end

function LuxCore.initialstates(rng::AbstractRNG, layer::Lux.ReactantLayer)
    st = LuxCore.initialstates(rng, layer.layer)
    layer.eltype_adaptor !== nothing && (st = adapt(layer.eltype_adaptor, st))
    return (; states=Lux.__make_reactant_array(st), training=Val(true))
end

function (l::Lux.ReactantLayer{FST, T})(x, ps, st::NamedTuple) where {FST, T}
    l.eltype_adaptor !== nothing && (x = adapt(l.eltype_adaptor, x))

    # XLARuntimeError is not great, so check and terminate early if needed
    @argcheck fmapstructure(Lux.__size, x) == l.input_structure

    # TODO: For non array inputs this we make the eltype uniform which might not be
    #       desirable. We should handle those cases with `fmap`
    if T != Lux.__recursive_eltype(x)
        @warn """
        `Reactant.compile` was called with input eltype $(T) but the current input eltype \
        is $(Lux.__recursive_eltype(x)). This might lead to unexpected behavior.

        We will convert the input to $(T) and continue. If you want to avoid this, please \
        recompile the model with the correct input eltype.
        """ maxlog=1
        x = adapt(LuxEltypeAdaptor{T}(), x)
    end

    return Lux.__apply_reactant(l, x, ps, st)
end

@inline function Lux.__apply_reactant(l::Lux.ReactantLayer, x, ps, st)
    y, st_ = Lux.__apply_reactant(l, x, ps, st.states, st.training)
    return y, (; states=st_, training=st.training)
end

# This is the ideal case where all the types match correctly.
# Input Type mispatches should not happen here, they should be handled before this function
# is called.
@inline function Lux.__apply_reactant(l::Lux.ReactantLayer{FST, T, inType}, x::inType,
        ps, st, training) where {FST, T, inType}
    return Lux.__apply_reactant(l, Lux.__make_reactant_array(x), ps, st, training)
end

@inline function Lux.__apply_reactant(
        l::Lux.ReactantLayer{FST, T, inType, inCType, stType, stTestType, psType},
        x::inCType, ps::psType, st::stType,
        training) where {FST, T, inType, inCType, psType, stType, stTestType}
    smodel = Lux.StatefulLuxLayer{FST}(l.layer, ps, st)
    return (
        Lux.__apply_reactant(l, smodel, x, training), ifelse(FST, smodel.st, smodel.st_any))
end

# Parameter type mismatch. This might be too common so try to handle it gracefully.
@inline function Lux.__apply_reactant(
        l::Lux.ReactantLayer{FST, T, inType, inCType, stType, stTestType, psType},
        x::inCType, ps::psType2, st,
        training) where {FST, T, inType, inCType, stType, stTestType, psType, psType2}
    @warn "Parameter Type Mismatch with compiled Reactant function. This will lead to \
           performance regressions" maxlog=1

    ps = __try_similar_structure(Lux.__named_tuple(ps), l.concrete_ps)
    ps = l.adaptor(ps)
    l.eltype_adaptor !== nothing && (ps = adapt(l.eltype_adaptor, ps))
    ps = Lux.__make_reactant_array(ps)

    if typeof(ps) != psType
        @warn "Automatic type conversion failed for `ps`." original_ps_type=psType2
        __graceful_type_mismatch_error(l, x, ps, st, training)
    end

    return Lux.__apply_reactant(l, Lux.__make_reactant_array(x), ps, st, training)
end

function Lux.__apply_reactant(l, x, ps, st, training)
    return __graceful_type_mismatch_error(l, x, ps, st, training)
end

@inline function Lux.__apply_reactant(l::Lux.ReactantLayer, smodel, x, ::Val{true})
    return l.fwd_fn(smodel, x)
end

@inline function Lux.__apply_reactant(l::Lux.ReactantLayer, smodel, x, ::Val{false})
    return l.inference_fn(smodel, x)
end

# Don't inline, else types don't get displayed in the stack trace
function __graceful_type_mismatch_error(
        ::Lux.ReactantLayer{FST, T, inType, inCType, stType, stTestType, psType},
        x,
        ps,
        st,
        ::Val{training}) where {
        FST, T, inType, inCType, psType, stType, stTestType, training}
    #! format: off
    input_type_mismatch_str = typeof(x) == inType || typeof(x) == inCType ? """
      1. Input Types Matched.
    """ : """
      1. Input Type: $(typeof(x)).
          Compiled Input Type: $(inType).
          Compiled Concrete Input Type: $(inCType).
    """
    #! format: on

    ps_type_mismatch_str = typeof(ps) == psType ? """
      2. Parameter Types Matched.
    """ : """
      2. Parameter Type: $(typeof(ps)).
          Compiled Parameter Type: $(psType).
    """

    st_type_mismatch_str = if training
        typeof(st) == stType ? """
          3. State Types Matched.
        """ : """
          3. State Type: $(typeof(st)).
              Compiled State Type: $(stType).
        """
    else
        typeof(st) == stTestType ? """
          3. State Types Matched.
        """ : """
          3. State Type: $(typeof(st)).
              Compiled State Type: $(stTestType).
        """
    end

    throw(ArgumentError("""
    Model compiled types and input types don't match. We tried our best to convert the \
    types to the right ones, but we failed. Ideally the argument types should not be \
    modified after compilation.

      1. Recompile the model with the correct input types.
      2. Open an issue on the Lux.jl repository, to check if we can ease out the automatic \
         type conversion.

    List of Type Mismatches:

     $(input_type_mismatch_str) $(ps_type_mismatch_str) $(st_type_mismatch_str)"""))
end

# TODO: For the iip versions as well. Metaprogram that part

# Case III: Nothing is cached. First call to `single_train_step`
function Lux.Experimental.single_train_step(
        ad::AutoReactant, obj_fn::F, data, ts::Lux.Experimental.TrainState) where {F}
    # ps = ts.parameters
    dps = Lux.__recursive_make_zero(ts.parameters)
    # st = ts.states
    # model = ts.model

    data = __make_concrete_array(data)
    model = __make_concrete_array(ts.model)
    dps = __make_concrete_array(dps)
    ps = __make_concrete_array(ts.parameters)
    st = __make_concrete_array(ts.states)

    # @show 

    # function reverse_fn_wrapper(obj_fn, model, ps, dps, st, data)
    obj_fn_wrapper, st_updated, stats = Lux.Experimental.__wrap_objective_function(
        obj_fn, st)
    # st_, stats = nothing, (;)
    # @show __update_fn_wrapper(obj_fn_wrapper, model, ps, dps, st, data)

    # function obj_fn_wrapper(obj_fn, model, ps, st, data) # Intentionally boxing
    #     y, st_, stats = obj_fn(model, ps, st, data)
    #     return y
    # end

    # @show obj_fn_wrapper # (obj_fn, model, ps, st, data)

    # _, loss = Enzyme.autodiff(Enzyme.ReverseWithPrimal, obj_fn_wrapper, Active,
    #     Const(model), Duplicated(ps, dps), Const(st), Const(data))
    # loss = obj_fn_wrapper(obj_fn, model, ps, st, data)

    # return loss, st_new, stats # FIXME: Return the correct things
    # return loss
    # end

    # @show reverse_fn_wrapper # (obj_fn, model, ps, dps, st, data)

    # @show reverse_fn_wrapper(obj_fn, model, ts.parameters,
    #     Lux.__recursive_make_zero(ts.parameters), ts.states, data)

    compiled_fn = Reactant.compile(__update_fn_wrapper, (obj_fn, model, ps, dps, st, data))

    # return compiled_fn, (obj_fn, model, ps, dps, st, data)
end

function __update_fn_wrapper(obj_fn, model, ps, dps, st, data)
    _, loss = Enzyme.autodiff(Enzyme.ReverseWithPrimal, obj_fn, Active, Const(model),
        Duplicated(ps, dps), Const(st), Const(data))
    # Lux.Experimental.apply_gradients()
    return loss
end

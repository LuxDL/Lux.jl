function Lux.Experimental.single_train_step!(
        backend::ReactantBackend, obj_fn::F, data, ts::TrainState) where {F}
    data = __make_reactant_array(data)
    ps = __make_reactant_array(ts.parameters)
    st = __make_reactant_array(ts.states)
    st_opt = __make_reactant_array(ts.optimizer_state)

    compiled_inference = if backend.input_prototype !== nothing
        Reactant.compile(LuxCore.apply,
            (ts.model, __make_reactant_array(backend.input_prototype),
                ps, LuxCore.testmode(st)))
    else
        nothing
    end

    compiled_grad_and_step! = Reactant.compile(
        internal_grad_and_step!, (obj_fn, ts.model, ps, st, st_opt, data, ts.optimizer))

    loss, st_updated, stats = compiled_grad_and_step!(
        obj_fn, ts.model, ps, st, st_opt, data, ts.optimizer)

    cache = TrainingBackendCache{:Reactant, false}(
        nothing, (; compiled_grad_and_step!, compiled_inference))
    ts_new = Lux.Experimental.TrainState(
        cache, obj_fn, ts.model, ps, st_updated, ts.optimizer, st_opt, ts.step + 1)

    return nothing, loss, stats, ts_new
end

function Lux.Experimental.single_train_step!(::ReactantBackend, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:Reactant}, F}) where {F}
    data = __make_reactant_array(data)

    loss, st_updated, stats = ts.cache.extras.compiled_grad_and_step!(
        obj_fn, ts.model, ts.parameters, ts.states, ts.optimizer_state, data, ts.optimizer)

    ts_new = Lux.Experimental.TrainState(
        ts.cache, obj_fn, ts.model, ts.parameters, st_updated,
        ts.optimizer, ts.optimizer_state, ts.step + 1)

    return nothing, loss, stats, ts_new
end

function internal_grad_and_step!(
        obj_fn::F, model, ps, st, st_opt, data, optimizer) where {F}
    dps = Lux.recursive_make_zero(ps)

    _, (loss, st_updated, stats) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, obj_fn, Active, Const(model),
        Duplicated(ps, dps), Const(st), Const(data))

    Lux.simple_optimizers_apply!(optimizer, st_opt, ps, dps) # ps & st_opt are updated in-place

    return loss, st_updated, stats
end

function (tstate::TrainState{<:TrainingBackendCache{:Reactant}})(data)
    data_reactant = __make_reactant_array(data)
    compiled_inference = if tstate.cache.extras.compiled_inference !== nothing
        tstate.cache.extras.compiled_inference
    else
        @warn "Inference function not compiled before. This will trigger compilation on \
               every inference call to `(::TrainState)(data)`. Please use \
               `ReactantBackend(; input_prototype = data)` to compile the inference \
               function on the first call to `single_train_step!` or \
               `single_train_step`." maxlog=1
        Reactant.compile(LuxCore.apply,
            (tstate.model, data_reactant, tstate.parameters, LuxCore.testmode(tstate.states)))
    end
    return compiled_inference(
        tstate.model, data_reactant, tstate.parameters, LuxCore.testmode(tstate.states))
end

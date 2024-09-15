function Lux.Training.single_train_step!(
        backend::ReactantBackend, obj_fn::F, data, ts::TrainState) where {F}
    data = Reactant.to_rarray(data)
    ps = Reactant.to_rarray(ts.parameters)
    st = Reactant.to_rarray(ts.states)
    st_opt = Reactant.to_rarray(ts.optimizer_state)

    compiled_inference = if backend.input_prototype !== nothing
        Reactant.compile(LuxCore.apply,
            (ts.model, Reactant.to_rarray(backend.input_prototype),
                ps, LuxCore.testmode(st)))
    else
        nothing
    end

    compiled_grad_and_step! = Reactant.compile(
        internal_grad_and_step!, (obj_fn, ts.model, ps, st, st_opt, data, ts.optimizer))

    loss, st_updated, stats = compiled_grad_and_step!(
        obj_fn, ts.model, ps, st, st_opt, data, ts.optimizer)

    cache = TrainingBackendCache(backend, False(), nothing, (; compiled_grad_and_step!,
        compiled_inference))
    @set! ts.cache = cache
    @set! ts.objective_function = obj_fn
    @set! ts.parameters = ps
    @set! ts.states = st_updated
    @set! ts.optimizer_state = st_opt
    @set! ts.step = ts.step + 1

    return nothing, loss, stats, ts # TODO: Return the gradients
end

function Lux.Training.single_train_step!(::ReactantBackend, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{<:ReactantBackend}, F}) where {F}
    data = Reactant.to_rarray(data)

    loss, st_updated, stats = ts.cache.extras.compiled_grad_and_step!(
        obj_fn, ts.model, ts.parameters, ts.states, ts.optimizer_state, data, ts.optimizer)

    @set! ts.objective_function = obj_fn
    @set! ts.states = st_updated
    @set! ts.step = ts.step + 1

    return nothing, loss, stats, ts # TODO: Return the gradients
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

function (tstate::TrainState{<:TrainingBackendCache{<:ReactantBackend}})(data)
    data_reactant = Reactant.to_rarray(data)
    compiled_inference = if tstate.cache.extras.compiled_inference !== nothing
        tstate.cache.extras.compiled_inference
    else
        @warn "Inference function not compiled before. This will trigger compilation on \
               every inference call to `(::TrainState)(data)`. Please use \
               `ReactantBackend(; input_prototype = data)` to compile the inference \
               function on the first call to `single_train_step!` or \
               `single_train_step`." maxlog=1
        Reactant.compile(LuxCore.apply,
            (tstate.model, data_reactant, tstate.parameters,
                LuxCore.testmode(tstate.states)))
    end
    return compiled_inference(
        tstate.model, data_reactant, tstate.parameters, LuxCore.testmode(tstate.states))
end

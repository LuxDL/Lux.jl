function Lux.Experimental.single_train_step!(
        ::AutoReactant, obj_fn::F, data, ts::TrainState) where {F}
    data = Lux.__make_reactant_array(data)
    ps = Lux.__make_reactant_array(ts.parameters)
    st = Lux.__make_reactant_array(ts.states)
    st_opt = Lux.__make_reactant_array(ts.optimizer_state)

    @info "First call to Reactant. Compiling the training function!" # TODO: Remove

    compiled_grad_and_step! = Reactant.compile(
        internal_grad_and_step!, (obj_fn, ts.model, ps, st, st_opt, data, ts.optimizer))

    loss, st_updated, stats = compiled_grad_and_step!(
        obj_fn, ts.model, ps, st, st_opt, data, ts.optimizer)

    cache = TrainingBackendCache{:Reactant, false}(nothing, (; compiled_grad_and_step!))
    ts_new = Lux.Experimental.TrainState(
        cache, obj_fn, ts.model, ps, st_updated, ts.optimizer, st_opt, ts.step + 1)

    return nothing, loss, stats, ts_new
end

function internal_grad_and_step!(
        obj_fn::F, model, ps, st, st_opt, data, optimizer) where {F}
    dps = Lux.recursive_make_zero(ps)

    _, (loss, st_updated, stats) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, obj_fn, Active, Const(model),
        Duplicated(ps, dps), Const(st), Const(data))

    simple_optimizers_apply!(optimizer, st_opt, ps, dps) # ps & st_opt are updated in-place

    return loss, st_updated, stats
end

function Lux.Experimental.single_train_step!(::AutoReactant, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:Reactant}, F}) where {F}
    data = Lux.__make_reactant_array(data)

    @info "We have already compiled the function!" # TODO: Remove

    loss, st_updated, stats = ts.cache.extras.compiled_grad_and_step!(
        obj_fn, ts.model, ts.parameters, ts.states, ts.optimizer_state, data, ts.optimizer)

    ts_new = Lux.Experimental.TrainState(
        ts.cache, obj_fn, ts.model, ts.parameters, st_updated,
        ts.optimizer, ts.optimizer_state, ts.step + 1)

    return nothing, loss, stats, ts_new
end

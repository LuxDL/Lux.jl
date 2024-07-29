function Lux.Training.compute_gradients(::AutoTracker, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:Tracker, FT}}) where {F, FT}
    dparams = FT ? ts.cache.dparameters : Lux.recursive_make_zero!!(ts.cache.dparameters)
    ps_tracked = __construct_tracked_params(ts.parameters, dparams)

    loss, st, stats = obj_fn(ts.model, ps_tracked, ts.states, data)
    Tracker.back!(loss)

    ts_new = TrainState(
        TrainingBackendCache{:Tracker, false}(ts.cache.dparameters, nothing), obj_fn,
        ts.model, ts.parameters, st, ts.optimizer, ts.optimizer_state, ts.step)

    return dparams, loss.data, stats, ts_new
end

function Lux.Training.compute_gradients(
        ::AutoTracker, obj_fn::F, data, ts::TrainState) where {F}
    grads = Lux.recursive_make_zero(ts.parameters)
    ts_new = TrainState(
        TrainingBackendCache{:Tracker, true}(grads, nothing), obj_fn, ts.model,
        ts.parameters, ts.states, ts.optimizer, ts.optimizer_state, ts.step)
    return Lux.Training.compute_gradients(AutoTracker(), obj_fn, data, ts_new)
end

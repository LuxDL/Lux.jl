function Lux.Training.compute_gradients_impl(
    ::AutoTracker, obj_fn::F, data, ts::TrainState{<:TrainingBackendCache{AutoTracker}}
) where {F}
    dps = Training.dparameters(ts.cache)
    ps_tracked = construct_tracked_params(ts.parameters, dps)

    loss, st, stats = obj_fn(ts.model, ps_tracked, ts.states, data)
    Tracker.back!(loss)

    @set! ts.cache.first_try = False()
    @set! ts.objective_function = obj_fn
    @set! ts.states = st

    return dps, loss.data, stats, ts
end

function Lux.Training.compute_gradients_impl(
    ad::AutoTracker, obj_fn::F, data, ts::TrainState
) where {F}
    grads = fmap(Utils.zero, ts.parameters; exclude=isleaf)
    cache = TrainingBackendCache(ad, True(), grads, nothing)
    @set! ts.cache = cache
    @set! ts.objective_function = obj_fn
    return Lux.Training.compute_gradients(ad, obj_fn, data, ts)
end

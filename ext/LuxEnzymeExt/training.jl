function Lux.Training.compute_gradients(
        ::AutoEnzyme, obj_fn::F, data, ts::TrainState) where {F}
    dps = Lux.recursive_make_zero(ts.parameters)

    obj_fn_wrap, st_wrap, stats_wrap = Lux.Training.__wrap_objective_function(
        obj_fn, ts.model, ts.parameters, ts.states, data, Val(true))

    _, loss = Enzyme.autodiff(
        EnzymeCore.ReverseWithPrimal, Const(obj_fn_wrap), Active, Const(ts.model),
        Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    cache = TrainingBackendCache{:Enzyme, false}(
        dps, (; obj_fn=obj_fn_wrap, st_wrap, stats_wrap))
    ts_new = TrainState(cache, obj_fn, ts.model, ts.parameters, st_wrap[],
        ts.optimizer, ts.optimizer_state, ts.step)

    return dps, loss, stats_wrap[], ts_new
end

const AUTODIFF_CACHE_TYPE = TrainingBackendCache{
    :Enzyme, false, PS, <:NamedTuple{(:obj_fn, :st_wrap, :stats_wrap)}} where {PS}

function Lux.Training.compute_gradients(
        ::AutoEnzyme, obj_fn::F, data, ts::TrainState{<:AUTODIFF_CACHE_TYPE, F}) where {F}
    dps = Lux.recursive_make_zero!!(ts.cache.dparameters)

    _, loss = Enzyme.autodiff(
        EnzymeCore.ReverseWithPrimal, Const(ts.cache.extras.obj_fn), Active,
        Const(ts.model), Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    ts_new = TrainState(
        ts.cache, obj_fn, ts.model, ts.parameters, ts.cache.extras.st_wrap[],
        ts.optimizer, ts.optimizer_state, ts.step)

    return dps, loss, ts.cache.extras.stats_wrap[], ts_new
end

function Lux.Training.compute_gradients(ad::AutoEnzyme, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:Enzyme, false}}) where {F}
    @warn "Detected calls to `compute_gradients(::AutoEnzyme, ...)` with objective \
           function that is changing across function calls. This can lead to the \
           generation of slow code" maxlog=1

    mode = EnzymeCore.ReverseModeSplit{
        true, true, 1, ntuple(Returns(false), 5), Enzyme.FFIABI}()
    forward, reverse = Enzyme.autodiff_thunk(
        mode, Const{typeof(obj_fn)}, Active, Const{typeof(ts.model)},
        Duplicated{typeof(ts.parameters)}, Const{typeof(ts.states)}, Const{typeof(data)})

    cache = TrainingBackendCache{:Enzyme, false}(ts.cache.dparameters, (; forward, reverse))
    ts_new = TrainState(cache, obj_fn, ts.model, ts.parameters, ts.states,
        ts.optimizer, ts.optimizer_state, ts.step)

    return Lux.Training.compute_gradients(ad, obj_fn, data, ts_new)
end

const AUTODIFF_THUNK_CACHE_TYPE = TrainingBackendCache{
    :Enzyme, false, PS, <:NamedTuple{(:forward, :reverse)}} where {PS}

function Lux.Training.compute_gradients(::AutoEnzyme, obj_fn::F, data,
        ts::TrainState{<:AUTODIFF_THUNK_CACHE_TYPE, F}) where {F}
    dps = Lux.recursive_make_zero!!(ts.cache.dparameters)
    params = Duplicated(ts.parameters, dps)

    tape, (loss, st_, stats), _ = ts.cache.extras.forward(
        Const(obj_fn), Const(ts.model), params, Const(ts.states), Const(data))
    ts.cache.extras.reverse(
        Const(obj_fn), Const(ts.model), params, Const(ts.states), Const(data),
        (one(loss), Lux.recursive_make_zero(st_), Lux.recursive_make_zero(stats)), tape)

    ts_new = TrainState(ts.cache, obj_fn, ts.model, ts.parameters, st_,
        ts.optimizer, ts.optimizer_state, ts.step)

    return dps, loss, stats, ts_new
end

function match_runtime_activity(ad_mode, actual_mode)
    if ad_mode !== nothing && EnzymeCore.runtime_activity(ad_mode)
        return EnzymeCore.set_runtime_activity(actual_mode)
    end
    return actual_mode
end

function Lux.Training.compute_gradients_impl(
    ad::AutoEnzyme, obj_fn::F, data, ts::TrainState
) where {F}
    dps = fmap(Utils.zero, ts.parameters; exclude=isleaf)

    obj_fn_wrap, st_wrap, stats_wrap = Lux.Training.wrap_objective_function(
        obj_fn, ts.model, ts.parameters, ts.states, data, True()
    )

    _, loss = Enzyme.autodiff(
        match_runtime_activity(ad.mode, EnzymeCore.ReverseWithPrimal),
        Const(obj_fn_wrap),
        Active,
        Const(ts.model),
        Duplicated(ts.parameters, dps),
        Const(ts.states),
        Const(data),
    )

    cache = TrainingBackendCache(
        ad, False(), dps, (; obj_fn=obj_fn_wrap, st_wrap, stats_wrap)
    )
    @set! ts.cache = cache
    @set! ts.objective_function = obj_fn
    @set! ts.states = st_wrap[]
    return dps, loss, stats_wrap[], ts
end

const AUTODIFF_CACHE_TYPE = TrainingBackendCache{
    <:AutoEnzyme,False,PS,<:NamedTuple{(:obj_fn, :st_wrap, :stats_wrap)}
} where {PS}

function Lux.Training.compute_gradients_impl(
    ad::AutoEnzyme, obj_fn::F, data, ts::TrainState{<:AUTODIFF_CACHE_TYPE,F}
) where {F}
    dps = Lux.Training.dparameters(ts.cache)

    _, loss = Enzyme.autodiff(
        match_runtime_activity(ad.mode, EnzymeCore.ReverseWithPrimal),
        Const(ts.cache.extras.obj_fn),
        Active,
        Const(ts.model),
        Duplicated(ts.parameters, dps),
        Const(ts.states),
        Const(data),
    )

    @set! ts.objective_function = obj_fn
    @set! ts.states = ts.cache.extras.st_wrap[]

    return dps, loss, ts.cache.extras.stats_wrap[], ts
end

function Lux.Training.compute_gradients_impl(
    ad::AutoEnzyme,
    obj_fn::F,
    data,
    ts::TrainState{<:TrainingBackendCache{<:AutoEnzyme,False}},
) where {F}
    @warn "Detected calls to `compute_gradients(::AutoEnzyme, ...)` with objective \
           function that is changing across function calls. This can lead to the \
           generation of slow code" maxlog = 1

    forward, reverse = Enzyme.autodiff_thunk(
        match_runtime_activity(ad.mode, EnzymeCore.ReverseSplitWithPrimal),
        Const{typeof(obj_fn)},
        Active,
        Const{typeof(ts.model)},
        Duplicated{typeof(ts.parameters)},
        Const{typeof(ts.states)},
        Const{typeof(data)},
    )

    cache = TrainingBackendCache(ad, False(), ts.cache.dparameters, (; forward, reverse))
    @set! ts.cache = cache
    @set! ts.objective_function = obj_fn
    return Lux.Training.compute_gradients(ad, obj_fn, data, ts)
end

const AUTODIFF_THUNK_CACHE_TYPE = TrainingBackendCache{
    <:AutoEnzyme,False,PS,<:NamedTuple{(:forward, :reverse)}
} where {PS}

function Lux.Training.compute_gradients_impl(
    ::AutoEnzyme, obj_fn::F, data, ts::TrainState{<:AUTODIFF_THUNK_CACHE_TYPE,F}
) where {F}
    dps = Lux.Training.dparameters(ts.cache)
    params = Duplicated(ts.parameters, dps)

    tape, (loss, st_, stats), _ = ts.cache.extras.forward(
        Const(obj_fn), Const(ts.model), params, Const(ts.states), Const(data)
    )
    ts.cache.extras.reverse(
        Const(obj_fn),
        Const(ts.model),
        params,
        Const(ts.states),
        Const(data),
        (
            one(loss),
            fmap(Utils.zero, st_; exclude=isleaf),
            fmap(Utils.zero, stats; exclude=isleaf),
        ),
        tape,
    )

    @set! ts.objective_function = obj_fn
    @set! ts.states = st_
    return dps, loss, stats, ts
end

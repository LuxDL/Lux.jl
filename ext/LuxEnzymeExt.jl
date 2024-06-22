module LuxEnzymeExt

using ADTypes: AutoEnzyme
using Enzyme: Enzyme, Active, Const, Duplicated
using Lux: Lux
using Lux.Experimental: TrainingBackendCache, TrainState

# Case I: We have TrainingBackendCache{:Enzyme} and obj_fn is unchanged.
function Lux.Experimental.compute_gradients(::AutoEnzyme, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:Enzyme, FT}, F}) where {F, FT}
    dps = FT ? ts.cache.dparameters : Lux.recursive_make_zero!!(ts.cache.dparameters)

    _, loss = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, ts.cache.extras.obj_fn, Active, Const(ts.model),
        Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    ts_new = __construct_new_trainstate(
        ts.cache.extras.st_wrap[], ts.states, ts, obj_fn, dps,
        ts.cache.extras.obj_fn, ts.cache.extras.st_wrap, ts.cache.extras.stats_wrap)

    return dps, loss, ts.cache.extras.stats_wrap[], ts_new
end

# Case II: We have CachedEnzymeExtras and obj_fn is changed.
function Lux.Experimental.compute_gradients(::AutoEnzyme, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:Enzyme, FT}}) where {F, FT}
    dps = FT ? ts.cache.dparameters : Lux.recursive_make_zero!!(ts.cache.dparameters)

    obj_fn_wrap, st_wrap, stats_wrap = Lux.Experimental.__wrap_objective_function(
        obj_fn, ts.model, ts.parameters, ts.states, data, Val(FT))

    _, loss = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, obj_fn_wrap, Active, Const(ts.model),
        Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    ts_new = __construct_new_trainstate(
        st_wrap[], ts.states, ts, obj_fn, dps, obj_fn_wrap, st_wrap, stats_wrap)

    return dps, loss, stats_wrap[], ts_new
end

# Case III: Nothing is cached. First call to `compute_gradients`
function Lux.Experimental.compute_gradients(
        ad::AutoEnzyme, obj_fn::F, data, ts::TrainState) where {F}
    dps = Lux.recursive_make_zero(ts.parameters)
    cache = TrainingBackendCache{:Enzyme, true}(
        dps, (; obj_fn=nothing, st_wrap=nothing, stats_wrap=nothing))
    ts_new = TrainState(
        cache, nothing, ts.model, ts.parameters, ts.states, ts.optimizer_state, ts.step)
    return Lux.Experimental.compute_gradients(ad, obj_fn, data, ts_new)
end

# If `st_new` is of a new type, we will have to recompute the cache anyway. Force it by not
# storing the objective function.
function __construct_new_trainstate(st_new::S, ::S, ts::TrainState, objective_fn::O, dps,
        obj_fn::O2, st_wrap, stats_wrap) where {S, O, O2}
    cache = TrainingBackendCache{:Enzyme, false}(dps, (; obj_fn, st_wrap, stats_wrap))
    return TrainState(
        cache, objective_fn, ts.model, ts.parameters, st_new, ts.optimizer_state, ts.step)
end

function __construct_new_trainstate(st_new, _, ts::TrainState, objective_fn::O, dps,
        obj_fn::O2, st_wrap, stats_wrap) where {O, O2}
    cache = TrainingBackendCache{:Enzyme, false}(dps, (; obj_fn, st_wrap, stats_wrap))
    return TrainState(
        cache, nothing, ts.model, ts.parameters, st_new, ts.optimizer_state, ts.step)
end

end

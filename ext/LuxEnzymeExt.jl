module LuxEnzymeExt

using ADTypes: AutoEnzyme
using Enzyme: Enzyme, Active, Const, Duplicated
using Lux: Lux
using Lux.Experimental: TrainingBackendCache, TrainState

# Case I: We have TrainingBackendCache{:Enzyme} and obj_fn is unchanged.
function Lux.Experimental.compute_gradients(::AutoEnzyme, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:Enzyme, false}, F}) where {F}
    dps = Lux.recursive_make_zero!!(ts.cache.dparameters)

    _, loss = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, ts.cache.extras.obj_fn, Active, Const(ts.model),
        Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    ts_new = TrainState(
        ts.cache, obj_fn, ts.model, ts.parameters, ts.cache.extras.st_wrap[],
        ts.optimizer, ts.optimizer_state, ts.step)

    return dps, loss, ts.cache.extras.stats_wrap[], ts_new
end

# Case II: We have CachedEnzymeExtras and obj_fn is changed.
# TODO: If not FT then use split primal mode which is not the most efficient.
function Lux.Experimental.compute_gradients(::AutoEnzyme, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:Enzyme, FT}}) where {F, FT}
    dps = FT ? ts.cache.dparameters : Lux.recursive_make_zero!!(ts.cache.dparameters)

    obj_fn_wrap, st_wrap, stats_wrap = Lux.Experimental.__wrap_objective_function(
        obj_fn, ts.model, ts.parameters, ts.states, data, Val(FT))

    _, loss = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, obj_fn_wrap, Active, Const(ts.model),
        Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    cache = TrainingBackendCache{:Enzyme, false}(
        dps, (; obj_fn=obj_fn_wrap, st_wrap, stats_wrap))
    ts_new = TrainState(cache, obj_fn, ts.model, ts.parameters, st_wrap[],
        ts.optimizer, ts.optimizer_state, ts.step)

    return dps, loss, stats_wrap[], ts_new
end

# Case III: Nothing is cached. First call to `compute_gradients`
function Lux.Experimental.compute_gradients(
        ad::AutoEnzyme, obj_fn::F, data, ts::TrainState) where {F}
    dps = Lux.recursive_make_zero(ts.parameters)
    cache = TrainingBackendCache{:Enzyme, true}(
        dps, (; obj_fn=nothing, st_wrap=nothing, stats_wrap=nothing))
    ts_new = TrainState(cache, nothing, ts.model, ts.parameters, ts.states,
        ts.optimizer, ts.optimizer_state, ts.step)
    return Lux.Experimental.compute_gradients(ad, obj_fn, data, ts_new)
end

end

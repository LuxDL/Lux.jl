module LuxEnzymeExt

using ADTypes: AutoEnzyme
using ConcreteStructs: @concrete
using Enzyme: Enzyme, Active, Const, Duplicated
using Lux: Lux

@concrete struct CachedEnzymeExtras{FT}
    dparameters
    objective_function
    st_wrap
    stats_wrap
end

# Case I: We have CachedEnzymeExtras and objective_function is unchanged.
function Lux.Experimental.compute_gradients(::AutoEnzyme, objective_function::F, data,
        ts::Lux.Experimental.TrainState{<:CachedEnzymeExtras{FT}, F}) where {F, FT}
    dps = Lux.recursive_make_zero!!(ts.cache.dparameters)

    _, loss = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, ts.cache.objective_function, Active, Const(ts.model),
        Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    ts_new = __construct_new_trainstate(
        ts.cache.st_wrap[], ts.states, ts, objective_function, dps,
        ts.cache.objective_function, ts.cache.st_wrap, ts.cache.stats_wrap)

    return dps, loss, ts.cache.stats_wrap[], ts_new
end

# Case II: We have CachedEnzymeExtras and objective_function is changed.
function Lux.Experimental.compute_gradients(::AutoEnzyme, objective_function::F, data,
        ts::Lux.Experimental.TrainState{<:CachedEnzymeExtras{FT}}) where {F, FT}
    dps = Lux.recursive_make_zero!!(ts.cache.dparameters)

    obj_fn, st_wrap, stats_wrap = Lux.Experimental.__wrap_objective_function(
        objective_function, ts.model, ts.parameters, ts.states, data, Val(FT))

    _, loss = Enzyme.autodiff(Enzyme.ReverseWithPrimal, obj_fn, Active, Const(ts.model),
        Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    ts_new = __construct_new_trainstate(
        st_wrap[], ts.states, ts, objective_function, dps, obj_fn, st_wrap, stats_wrap)

    return dps, loss, stats_wrap[], ts_new
end

# Case III: Nothing is cached. First call to `compute_gradients`
function Lux.Experimental.compute_gradients(ad::AutoEnzyme, objective_function::F, data,
        ts::Lux.Experimental.TrainState) where {F}
    dps = Lux.recursive_make_zero(ts.parameters)
    cache = CachedEnzymeExtras{true}(dps, nothing, nothing, nothing)
    ts_new = Lux.Experimental.TrainState(
        cache, nothing, ts.model, ts.parameters, ts.states, ts.optimizer_state, ts.step)
    return Lux.Experimental.compute_gradients(ad, objective_function, data, ts_new)
end

# If `st_new` is of a new type, we will have to recompute the cache anyway. Force it by not
# storing the objective function.
function __construct_new_trainstate(
        st_new::S, ::S, ts::Lux.Experimental.TrainState, objective_fn::O,
        dps, obj_fn::O2, st_wrap, stats_wrap) where {S, O, O2}
    cache = CachedEnzymeExtras{false}(dps, obj_fn, st_wrap, stats_wrap)
    return Lux.Experimental.TrainState(
        cache, objective_fn, ts.model, ts.parameters, st_new, ts.optimizer_state, ts.step)
end

function __construct_new_trainstate(
        st_new, _, ts::Lux.Experimental.TrainState, objective_fn::O,
        dps, obj_fn::O2, st_wrap, stats_wrap) where {O, O2}
    cache = CachedEnzymeExtras{false}(dps, nothing, nothing, nothing)
    return Lux.Experimental.TrainState(
        cache, nothing, ts.model, ts.parameters, st_new, ts.optimizer_state, ts.step)
end

end

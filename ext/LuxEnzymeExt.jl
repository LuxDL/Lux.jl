module LuxEnzymeExt

using ADTypes: AutoEnzyme
using ConcreteStructs: @concrete
using Enzyme: Enzyme, Active, Const, Duplicated
using Lux: Lux
using Setfield: @set!

@concrete struct CachedEnzymeExtras
    dparameters
    objective_function
    st_wrap
    stats_wrap
end

# Case I: We have CachedEnzymeExtras and objective_function is unchanged.
function Lux.Experimental.compute_gradients(::AutoEnzyme, objective_function::F, data,
        ts::Lux.Experimental.TrainState{<:CachedEnzymeExtras, F}) where {F}
    dps = ts.cache.dparameters
    Lux.__recursive_make_zero!(dps)

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
        ts::Lux.Experimental.TrainState{<:CachedEnzymeExtras}) where {F}
    dps = ts.cache.dparameters
    Lux.__recursive_make_zero!(dps)

    obj_fn, st_wrap, stats_wrap = Lux.Experimental.__wrap_objective_function(
        objective_function, ts.model, ts.parameters, ts.states, data)

    _, loss = Enzyme.autodiff(Enzyme.ReverseWithPrimal, obj_fn, Active, Const(ts.model),
        Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    ts_new = __construct_new_trainstate(
        st_wrap[], ts.states, ts, objective_function, dps, obj_fn, st_wrap, stats_wrap)

    return dps, loss, stats_wrap[], ts_new
end

# Case III: Nothing is cached. First call to `compute_gradients`
function Lux.Experimental.compute_gradients(::AutoEnzyme, objective_function::F, data,
        ts::Lux.Experimental.TrainState) where {F}
    dps = Lux.__recursive_make_zero(ts.parameters)

    obj_fn, st_wrap, stats_wrap = Lux.Experimental.__wrap_objective_function(
        objective_function, ts.model, ts.parameters, ts.states, data)

    _, loss = Enzyme.autodiff(Enzyme.ReverseWithPrimal, obj_fn, Active, Const(ts.model),
        Duplicated(ts.parameters, dps), Const(ts.states), Const(data))

    ts_new = __construct_new_trainstate(
        st_wrap[], ts.states, ts, objective_function, dps, obj_fn, st_wrap, stats_wrap)

    return dps, loss, stats_wrap[], ts_new
end

# If `st_new` is of a new type, we will have to recompute the cache anyway. Force it by not
# storing the objective function.
function __construct_new_trainstate(
        st_new::S, ::S, ts::Lux.Experimental.TrainState, objective_fn::O,
        dps, obj_fn::O2, st_wrap, stats_wrap) where {S, O, O2}
    cache = CachedEnzymeExtras(dps, obj_fn, st_wrap, stats_wrap)
    return Lux.Experimental.TrainState(
        cache, objective_fn, ts.model, ts.parameters, st_new, ts.optimizer_state, ts.step)
end

function __construct_new_trainstate(
        st_new, _, ts::Lux.Experimental.TrainState, objective_fn::O,
        dps, obj_fn::O2, st_wrap, stats_wrap) where {O, O2}
    cache = CachedEnzymeExtras(dps, nothing, nothing, nothing)
    return Lux.Experimental.TrainState(
        cache, nothing, ts.model, ts.parameters, st_new, ts.optimizer_state, ts.step)
end

end

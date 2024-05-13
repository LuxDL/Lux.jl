module LuxEnzymeExt

using ADTypes: AutoEnzyme
using ConcreteStructs: @concrete
using Enzyme: Enzyme, Active, Const, Duplicated
using Lux: Lux
using Setfield: @set!

@concrete struct CachedEnzymeExtras
    dparameters
    forward
    reverse
end

# Case I: We have CachedEnzymeExtras and objective_function is unchanged.
function Lux.Experimental.compute_gradients(::AutoEnzyme, objective_function::F, data,
        ts::Lux.Experimental.TrainState{<:CachedEnzymeExtras, F}) where {F}
    Lux.__recursive_make_zero!(ts.cache.dparameters)
    loss, st_new, stats = __compute_gradients!(
        ts.cache.forward, ts.cache.reverse, objective_function,
        ts.model, ts.parameters, ts.cache.dparameters, ts.states, data)
    ts_new = __construct_new_trainstate(
        st_new, ts.states, ts.cache.forward, ts.cache.reverse,
        ts, objective_function, ts.cache.dparameters)
    return ts.cache.dparameters, loss, stats, ts_new
end

# Case II: We have CachedEnzymeExtras and objective_function is changed.
function Lux.Experimental.compute_gradients(::AutoEnzyme, objective_function::F, data,
        ts::Lux.Experimental.TrainState{<:CachedEnzymeExtras}) where {F}
    forward, reverse = Enzyme.autodiff_thunk(
        Enzyme.ReverseSplitWithPrimal, Const{typeof(objective_function)},
        Active, Const{typeof(ts.model)}, Duplicated{typeof(ts.parameters)},
        Const{typeof(ts.states)}, Const{typeof(data)})

    Lux.__recursive_make_zero!(ts.cache.dparameters)
    loss, st_new, stats = __compute_gradients!(
        forward, reverse, objective_function, ts.model,
        ts.parameters, ts.cache.dparameters, ts.states, data)

    ts_new = __construct_new_trainstate(
        st_new, ts.states, forward, reverse, ts, objective_function, ts.cache.dparameters)
    return ts.cache.dparameters, loss, stats, ts_new
end

# Case III: Nothing is cached
function Lux.Experimental.compute_gradients(::AutoEnzyme, objective_function::F, data,
        ts::Lux.Experimental.TrainState) where {F}
    dps = Lux.__recursive_make_zero(ts.parameters)
    forward, reverse = Enzyme.autodiff_thunk(
        Enzyme.ReverseSplitWithPrimal, Const{typeof(objective_function)},
        Active, Const{typeof(ts.model)}, Duplicated{typeof(ts.parameters)},
        Const{typeof(ts.states)}, Const{typeof(data)})
    loss, st_new, stats = __compute_gradients!(
        forward, reverse, objective_function, ts.model, ts.parameters, dps, ts.states, data)
    ts_new = __construct_new_trainstate(
        st_new, ts.states, forward, reverse, ts, objective_function, dps)
    return dps, loss, stats, ts_new
end

function __compute_gradients!(
        forward::F, reverse::R, obj_fn::O, model, ps, dps, st, data) where {F, R, O}
    pps = Duplicated(ps, dps)
    args = (Const(obj_fn), Const(model), pps, Const(st), Const(data))
    tape, (loss, st_new, stats), shadow_result = forward(args...)
    reverse(args...,
        (one(loss), Lux.__recursive_make_zero(st_new), Lux.__recursive_make_zero(stats)),
        tape)
    return loss, st_new, stats
end

# If `st_new` is of a new type, we will have to recompute the cache anyway. Force it by not
# storing the objective function.
function __construct_new_trainstate(
        st_new::S, ::S, forward::F, reverse::R, ts::Lux.Experimental.TrainState,
        objective_fn::O, dps) where {S, F, R, O}
    cache = CachedEnzymeExtras(dps, forward, reverse)
    return Lux.Experimental.TrainState(cache, objective_fn, ts.model, ts.parameters,
        st_new, ts.optimizer_state, ts.step + 1)
end

function __construct_new_trainstate(
        st_new, _, forward::F, reverse::R, ts::Lux.Experimental.TrainState,
        objective_fn::O, dps) where {F, R, O}
    cache = CachedEnzymeExtras(dps, nothing, nothing)
    return Lux.Experimental.TrainState(
        cache, nothing, ts.model, ts.parameters, st_new, ts.optimizer_state, ts.step + 1)
end

end

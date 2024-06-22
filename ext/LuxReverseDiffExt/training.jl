@static if pkgversion(ADTypes) < v"1.5"
    # older versions did not have `compile` type parameter. Use slower type-unstable code
    function Lux.Experimental.compute_gradients(
            ad::AutoReverseDiff, obj_fn::F, data, ts::TrainState) where {F}
        ad.compile && return __compiled_reverse_diff(obj_fn, data, ts)
        return __uncompiled_reverse_diff(obj_fn, data, ts)
    end
else
    for compiled in (false, true)
        fname = compiled ? :__compiled_reverse_diff : :__uncompiled_reverse_diff
        @eval function Lux.Experimental.compute_gradients(
                ::AutoReverseDiff{$(compiled)}, obj_fn::F, data, ts::TrainState) where {F}
            return $(fname)(obj_fn, data, ts)
        end
    end
end

# Uncompiled ReverseDiff
@inline function __uncompiled_reverse_diff(
        obj_fn::F, data, ts::TrainState{<:TrainingBackendCache{:ReverseDiff}}) where {F}
    tape = ReverseDiff.InstructionTape()
    ps_tracked = Lux.recursive_map(
        Lux.__Fix3(ReverseDiff.TrackedArray, tape), ts.parameters, ts.cache.dparameters)

    loss, st, stats = obj_fn(ts.model, ps_tracked, ts.states, data)
    loss.deriv = true
    ReverseDiff.reverse_pass!(tape)

    ts_new = TrainState(
        TrainingBackendCache{:ReverseDiff, false}(ts.cache.dparameters, obj_fn, nothing),
        obj_fn, ts.model, ts.parameters, st, ts.optimizer_state, ts.step)

    return ts.cache.dparameters, ReverseDiff.value(loss), stats, ts_new
end

# First call, nothing is cached
@inline function __uncompiled_reverse_diff(obj_fn::F, data, ts::TrainState) where {F}
    grads = Lux.recursive_make_zero(ts.parameters)
    ts_new = TrainState(TrainingBackendCache{:ReverseDiff, true}(grads, obj_fn, nothing),
        obj_fn, ts.model, ts.parameters, ts.states, ts.optimizer_state, ts.step)
    return __uncompiled_reverse_diff(obj_fn, data, ts_new)
end

# Compiled ReverseDiff
@inline function __compiled_reverse_diff(obj_fn::F, data, ts::TrainState) where {F}
    error("Not implemented yet")
end

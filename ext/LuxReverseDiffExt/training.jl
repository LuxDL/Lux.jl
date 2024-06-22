@static if pkgversion(ADTypes) < v"1.5"
    # older versions did not have `compile` type parameter. Use slower type-unstable code
    function Lux.Experimental.compute_gradients(ad::AutoReverseDiff, objective_function::F,
            data, ts::Lux.Experimental.TrainState) where {F}
        ad.compile && return __compiled_reverse_diff(objective_function, data, ts)
        return __uncompiled_reverse_diff(objective_function, data, ts)
    end
else
    for compiled in (false, true)
        fname = compiled ? :__compiled_reverse_diff : :__uncompiled_reverse_diff
        @eval function Lux.Experimental.compute_gradients(
                ::AutoReverseDiff{$(compiled)}, objective_function::F,
                data, ts::Lux.Experimental.TrainState) where {F}
            return $(fname)(objective_function, data, ts)
        end
    end
end

@inline function __uncompiled_reverse_diff(
        objective_function::F, data, ts::Lux.Experimental.TrainState) where {F}
    tape = ReverseDiff.InstructionTape()
    grads = Lux.recursive_make_zero(ts.parameters)
    ps_tracked = Lux.recursive_map(
        Lux.__Fix3(ReverseDiff.TrackedArray, tape), ts.parameters, grads)
    loss, st, stats = objective_function(ts.model, ps_tracked, ts.states, data)
    loss.deriv = true
    ReverseDiff.reverse_pass!(tape)
    @set! ts.states = st
    return grads, ReverseDiff.value(loss), stats, ts
end

@inline function __compiled_reverse_diff(
        objective_function::F, data, ts::Lux.Experimental.TrainState) where {F}
    # tape = ReverseDiff.InstructionTape()
    # grads = Lux.recursive_make_zero(ts.parameters)
    # ps_tracked = Lux.recursive_map(
    #     Lux.__Fix3(ReverseDiff.TrackedArray, tape), ts.parameters, grads)
    # loss, st, stats = objective_function(ts.model, ps_tracked, ts.states, data)
    # loss.deriv = true
    # ReverseDiff.reverse_pass!(tape)
    # @set! ts.states = st
    # return grads, ReverseDiff.value(loss), stats, ts
    error("Not implemented yet")
end

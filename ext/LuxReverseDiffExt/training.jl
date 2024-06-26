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
@inline function __uncompiled_reverse_diff(obj_fn::F, data, ts::TrainState) where {F}
    grads = Lux.recursive_make_zero(ts.parameters)
    ts_new = TrainState(
        TrainingBackendCache{:ReverseDiff, true}(grads, nothing), obj_fn, ts.model,
        ts.parameters, ts.states, ts.optimizer, ts.optimizer_state, ts.step)
    return __uncompiled_reverse_diff(obj_fn, data, ts_new)
end

@inline function __uncompiled_reverse_diff(obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:ReverseDiff, FT}}) where {F, FT}
    dparams = FT ? ts.cache.dparameters : Lux.recursive_make_zero!!(ts.cache.dparameters)
    tape = ReverseDiff.InstructionTape()
    ps_tracked = Lux.recursive_map(
        Lux.__Fix3(ReverseDiff.TrackedArray, tape), ts.parameters, dparams)

    loss, st, stats = obj_fn(ts.model, ps_tracked, ts.states, data)
    loss.deriv = true
    ReverseDiff.reverse_pass!(tape)

    ts_new = TrainState(
        TrainingBackendCache{:ReverseDiff, false}(ts.cache.dparameters, nothing),
        obj_fn, ts.model, ts.parameters, st, ts.optimizer, ts.optimizer_state, ts.step)

    return ts.cache.dparameters, ReverseDiff.value(loss), stats, ts_new
end

# Compiled ReverseDiff
@inline function __compiled_reverse_diff(obj_fn::F, data, ts::TrainState) where {F}
    grads = Lux.recursive_make_zero(ts.parameters)
    data_cache = deepcopy(data)
    ps_cache = deepcopy(ts.parameters)
    extras = (; data_cache, ps_cache)

    ts_new = TrainState(
        TrainingBackendCache{:ReverseDiff, true}(grads, extras), nothing, ts.model,
        ts.parameters, ts.states, ts.optimizer, ts.optimizer_state, ts.step)
    return __compiled_reverse_diff(obj_fn, data, ts_new)
end

## Tape hasn't been compiled yet / Function mismatch so recompile
@inline function __compiled_reverse_diff(obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:ReverseDiff, FT}}) where {F, FT}
    if Lux.statelength(ts.states) != 0
        throw(ArgumentError("AutoReverseDiff(; compile=true) is not supported for Lux \
                             models with non-empty state `st`."))
    end

    if FT # do a dry run
        _, st_, stats = obj_fn(ts.model, ts.parameters, ts.states, data)
        if stats != NamedTuple()
            throw(ArgumentError("AutoReverseDiff(; compile=true) is not supported for \
                                 loss functions that return non-empty `stats`."))
        end
        if Lux.statelength(st_) != 0
            throw(ArgumentError("AutoReverseDiff(; compile=true) is not supported for \
                                 models with non-empty state `st`."))
        end
    end

    dparams = FT ? ts.cache.dparameters : Lux.recursive_make_zero!!(ts.cache.dparameters)

    (; ps_cache, data_cache) = ts.cache.extras
    if !FT
        Lux.recursive_copyto!(ps_cache, ts.parameters)
        Lux.recursive_copyto!(data_cache, data)
    end

    obj_fn_wrap = first ∘ obj_fn

    tape = ReverseDiff.InstructionTape()
    ps_tracked = Lux.recursive_map(
        Lux.__Fix3(ReverseDiff.TrackedArray, tape), ps_cache, dparams)

    loss = obj_fn_wrap(ts.model, ps_tracked, ts.states, data_cache)
    loss.deriv = true
    ReverseDiff.reverse_pass!(tape)

    forward_executor = [ReverseDiff.FunctionWrapper{Nothing, Tuple{}}(ReverseDiff.ForwardExecutor(instruction))
                        for instruction in tape]
    reverse_executor = [ReverseDiff.FunctionWrapper{Nothing, Tuple{}}(ReverseDiff.ReverseExecutor(tape[i]))
                        for i in length(tape):-1:1]

    compiled_extras = (;
        ps_cache, data_cache, forward_executor, reverse_executor, output=loss)
    ts_new = TrainState(
        TrainingBackendCache{:ReverseDiff, false}(ts.cache.dparameters, compiled_extras),
        obj_fn, ts.model, ts.parameters, ts.states,
        ts.optimizer, ts.optimizer_state, ts.step)

    return dparams, ReverseDiff.value(loss), NamedTuple(), ts_new
end

@inline function __compiled_reverse_diff(obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:ReverseDiff, false}, F}) where {F}
    (; ps_cache, data_cache, output) = ts.cache.extras

    dparams = Lux.recursive_make_zero!!(ts.cache.dparameters)
    Lux.recursive_copyto!(ps_cache, ts.parameters)
    Lux.recursive_copyto!(data_cache, data)

    for wrapper in ts.cache.extras.forward_executor
        wrapper()
    end
    output.deriv = true
    for wrapper in ts.cache.extras.reverse_executor
        wrapper()
    end

    ts_new = TrainState(ts.cache, obj_fn, ts.model, ts.parameters, ts.states,
        ts.optimizer, ts.optimizer_state, ts.step)
    return dparams, ReverseDiff.value(output), NamedTuple(), ts_new
end

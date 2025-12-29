# Uncompiled ReverseDiff
function Lux.Training.compute_gradients_impl(
    ad::AutoReverseDiff{false}, obj_fn::F, data, ts::TrainState
) where {F}
    @set! ts.cache = TrainingBackendCache(
        ad, True(), fmap(Utils.zero, ts.parameters; exclude=isleaf), nothing
    )
    @set! ts.objective_function = obj_fn
    return Lux.Training.compute_gradients(ad, obj_fn, data, ts)
end

function Lux.Training.compute_gradients_impl(
    ::AutoReverseDiff{false},
    obj_fn::F,
    data,
    ts::TrainState{<:TrainingBackendCache{AutoReverseDiff{false}}},
) where {F}
    dparams = Training.dparameters(ts.cache)
    tape = ReverseDiff.InstructionTape()
    ps_tracked = fmap(
        Utils.Fix3(TrackedArray, tape), ts.parameters, dparams; exclude=isleaf
    )

    loss, st, stats = obj_fn(ts.model, ps_tracked, ts.states, data)
    loss.deriv = true
    ReverseDiff.reverse_pass!(tape)

    @set! ts.cache.first_try = False()
    @set! ts.objective_function = obj_fn
    @set! ts.states = st
    return dparams, ReverseDiff.value(loss), stats, ts
end

# Compiled ReverseDiff
function Lux.Training.compute_gradients_impl(
    ad::AutoReverseDiff{true}, obj_fn::F, data, ts::TrainState
) where {F}
    @set! ts.cache = TrainingBackendCache(
        ad,
        True(),
        fmap(Utils.zero, ts.parameters; exclude=isleaf),
        (; data_cache=deepcopy(data), ps_cache=deepcopy(ts.parameters)),
    )
    @set! ts.objective_function = nothing

    return Lux.Training.compute_gradients(ad, obj_fn, data, ts)
end

## Tape hasn't been compiled yet / Function mismatch so recompile
function Lux.Training.compute_gradients_impl(
    ad::AutoReverseDiff{true},
    obj_fn::F,
    data,
    ts::TrainState{<:TrainingBackendCache{AutoReverseDiff{true}}},
) where {F}
    if LuxCore.statelength(ts.states) != 0
        throw(ArgumentError("AutoReverseDiff(; compile=true) is not supported for Lux \
                             models with non-empty state `st`."))
    end

    first_try = ts.cache.first_try isa True

    if first_try # do a dry run
        _, st_, stats = obj_fn(ts.model, ts.parameters, ts.states, data)
        if stats != NamedTuple()
            throw(ArgumentError("AutoReverseDiff(; compile=true) is not supported for \
                                 loss functions that return non-empty `stats`."))
        end
        if LuxCore.statelength(st_) != 0
            throw(ArgumentError("AutoReverseDiff(; compile=true) is not supported for \
                                 models with non-empty state `st`."))
        end
    end

    dparams = Training.dparameters(ts.cache)

    (; ps_cache, data_cache) = ts.cache.extras
    if !first_try
        fmap(copyto!, ps_cache, ts.parameters; exclude=isleaf)
        fmap(copyto!, data_cache, data; exclude=isleaf)
    end

    tape = ReverseDiff.InstructionTape()
    ps_tracked = fmap(Utils.Fix3(TrackedArray, tape), ps_cache, dparams; exclude=isleaf)

    loss = first(obj_fn(ts.model, ps_tracked, ts.states, data_cache))
    loss.deriv = true
    ReverseDiff.reverse_pass!(tape)

    forward_executor = [
        FunctionWrapper{Nothing,Tuple{}}(ForwardExecutor(instruction)) for
        instruction in tape
    ]
    reverse_executor = [
        FunctionWrapper{Nothing,Tuple{}}(ReverseExecutor(tape[i])) for
        i in length(tape):-1:1
    ]

    @set! ts.cache = TrainingBackendCache(
        ad,
        False(),
        dparams,
        (; ps_cache, data_cache, forward_executor, reverse_executor, output=loss),
    )
    @set! ts.objective_function = obj_fn
    return dparams, ReverseDiff.value(loss), NamedTuple(), ts
end

function Lux.Training.compute_gradients_impl(
    ::AutoReverseDiff{true},
    obj_fn::F,
    data,
    ts::TrainState{<:TrainingBackendCache{AutoReverseDiff{true}},F},
) where {F}
    (; ps_cache, data_cache, output) = ts.cache.extras

    dparams = Training.dparameters(ts.cache)
    fmap(copyto!, ps_cache, ts.parameters; exclude=isleaf)
    fmap(copyto!, data_cache, data; exclude=isleaf)

    for wrapper in ts.cache.extras.forward_executor
        wrapper()
    end
    output.deriv = true
    for wrapper in ts.cache.extras.reverse_executor
        wrapper()
    end

    @set! ts.cache.first_try = False()
    @set! ts.objective_function = obj_fn
    return dparams, ReverseDiff.value(output), NamedTuple(), ts
end

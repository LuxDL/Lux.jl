using ADTypes: AutoForwardDiff
using DiffResults: DiffResults
using ForwardDiff: ForwardDiff
using Setfield: @set!
using Static: True, False

function Training.compute_gradients_impl(
    ad::AutoForwardDiff, obj_fn::F, data, ts::Training.TrainState
) where {F}
    @assert ts.parameters isa AbstractArray "AutoForwardDiff only supports AbstractArray \
                                             parameters, not $(typeof(ts.parameters)). To \
                                             convert the parameter structure to an array \
                                             use `ComponentArray(ps)`."

    obj_fn_wrap, st_wrap, stats_wrap = Training.wrap_objective_function(
        obj_fn, ts.model, ts.parameters, ts.states, data, True()
    )

    gradient_result = DiffResults.GradientResult(ts.parameters)
    ForwardDiff.gradient!(
        gradient_result, ps -> obj_fn_wrap(ts.model, ps, ts.states, data), ts.parameters
    )

    cache = Training.TrainingBackendCache(
        ad, False(), gradient_result, (; obj_fn=obj_fn_wrap, st_wrap, stats_wrap)
    )
    @set! ts.cache = cache
    @set! ts.objective_function = obj_fn
    @set! ts.states = st_wrap[]
    return (
        DiffResults.gradient(gradient_result),
        DiffResults.value(gradient_result),
        stats_wrap[],
        ts,
    )
end

const FORWARDDIFF_CACHE_TYPE = Training.TrainingBackendCache{
    <:AutoForwardDiff,False,PS,<:NamedTuple{(:obj_fn, :st_wrap, :stats_wrap)}
} where {PS}

function Training.compute_gradients_impl(
    ::AutoForwardDiff, obj_fn::F, data, ts::Training.TrainState{<:FORWARDDIFF_CACHE_TYPE,F}
) where {F}
    gradient_result = ts.cache.dparameters

    ForwardDiff.gradient!(
        gradient_result,
        ps -> ts.cache.extras.obj_fn(ts.model, ps, ts.states, data),
        ts.parameters,
    )

    @set! ts.objective_function = obj_fn
    @set! ts.states = ts.cache.extras.st_wrap[]

    return (
        DiffResults.gradient(gradient_result),
        DiffResults.value(gradient_result),
        ts.cache.extras.stats_wrap[],
        ts,
    )
end

function Training.compute_gradients_impl(
    ::AutoForwardDiff,
    obj_fn::F,
    data,
    ts::Training.TrainState{<:Training.TrainingBackendCache{<:AutoForwardDiff,False}},
) where {F}
    @warn "Detected calls to `compute_gradients(::AutoForwardDiff, ...)` with objective \
           function that is changing across function calls. This can lead to the \
           generation of slow code" maxlog = 1
    gradient_result = ts.cache.dparameters

    # We do exactly same thing as the first case but without caching the function
    obj_fn_wrap, st_wrap, stats_wrap = Training.wrap_objective_function(
        obj_fn, ts.model, ts.parameters, ts.states, data, False()
    )

    ForwardDiff.gradient!(
        gradient_result, ps -> obj_fn_wrap(ts.model, ps, ts.states, data), ts.parameters
    )

    @set! ts.states = st_wrap[]
    return (
        DiffResults.gradient(gradient_result),
        DiffResults.value(gradient_result),
        stats_wrap[],
        ts,
    )
end

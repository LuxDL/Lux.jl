function Lux.Training.compute_gradients_impl(
    ad::AutoMooncake, objective_function::F, data, ts::TrainState
) where {F}
    config = get_config(ad)
    pullback_cache = prepare_pullback_cache(
        objective_function,
        ts.model,
        ts.parameters,
        ts.states,
        data;
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    # evaluate once to get the correct types
    loss, stₙ, stats = objective_function(ts.model, ts.parameters, ts.states, data)

    @set! ts.cache = TrainingBackendCache(
        ad,
        True(),
        nothing,
        (;
            pullback_cache,
            tangent_data=(
                one(loss), Mooncake.zero_tangent(stₙ), Mooncake.zero_tangent(stats)
            ),
        ),
    )
    @set! ts.objective_function = objective_function
    return Lux.Training.compute_gradients(ad, objective_function, data, ts)
end

function Lux.Training.compute_gradients_impl(
    ::AutoMooncake,
    objective_function::F,
    data,
    ts::TrainState{<:TrainingBackendCache{<:AutoMooncake},F},
) where {F}
    (loss, st, stats), (_, _, ∂ps, _, _) = value_and_pullback!!(
        ts.cache.extras.pullback_cache,
        ts.cache.extras.tangent_data,
        objective_function,
        ts.model,
        ts.parameters,
        ts.states,
        data,
    )

    @set! ts.states = st
    return ∂ps, loss, stats, ts
end

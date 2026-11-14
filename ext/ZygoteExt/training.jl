function Lux.Training.compute_gradients_impl(
    ::AutoZygote, objective_function::F, data, ts::Lux.Training.TrainState
) where {F}
    # Zygote 0.7 doesn't aggressively unthunk everything, so it is better to use a
    # closure here
    (loss, st, stats), back = Zygote.pullback(
        ps -> objective_function(ts.model, ps, ts.states, data), ts.parameters
    )
    grads = only(back((one(loss), nothing, nothing)))
    @set! ts.states = st
    return grads, loss, stats, ts
end

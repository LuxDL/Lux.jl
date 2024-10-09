function Lux.Training.compute_gradients_impl(
        ::AutoZygote, objective_function::F, data, ts::Lux.Training.TrainState) where {F}
    (loss, st, stats), back = Zygote.pullback(
        objective_function, ts.model, ts.parameters, ts.states, data)
    grads = back((one(loss), nothing, nothing))[2]
    @set! ts.states = st
    return grads, loss, stats, ts
end

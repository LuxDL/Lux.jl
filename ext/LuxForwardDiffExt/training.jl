function Lux.Training.compute_gradients_impl(
    ::AutoForwardDiff, objective_function::F, data, ts::Lux.Training.TrainState
) where {F}
    grads = gradient(ps -> objective_function(ts.model,ps,ts.states,data)[1], ts.parameters);
    loss, st, stats = objective_function(ts.model,ts.parameters,ts.states,data)
    @set! ts.states = st
    return grads, loss, stats, ts
end

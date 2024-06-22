function Lux.Experimental.compute_gradients(::AutoReverseDiff, objective_function::F, data,
        ts::Lux.Experimental.TrainState) where {F}
    tape = ReverseDiff.InstructionTape()
    grads = fmap(zero, ts.parameters)
    ps_tracked = fmap((p, g) -> ReverseDiff.TrackedArray(p, g, tape), ts.parameters, grads)
    loss, st, stats = objective_function(ts.model, ps_tracked, ts.states, data)
    loss.deriv = true
    ReverseDiff.reverse_pass!(tape)
    @set! ts.states = st
    return grads, ReverseDiff.value(loss), stats, ts
end

module LuxZygoteExt

using ADTypes, Lux, Setfield, Zygote
using TruncatedStacktraces: @truncate_stacktrace
using Zygote: Pullback

function Lux.Training.compute_gradients(::AutoZygote, objective_function::Function, data,
    ts::Lux.Training.TrainState)
    (loss, st, stats), back = Zygote.pullback(ps -> objective_function(ts.model, ps,
            ts.states, data), ts.parameters)
    grads = back((one(loss), nothing, nothing))[1]
    @set! ts.states = st
    return grads, loss, stats, ts
end

@truncate_stacktrace Pullback 1

end

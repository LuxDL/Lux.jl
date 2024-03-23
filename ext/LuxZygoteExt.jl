module LuxZygoteExt

using ADTypes, Lux, Setfield, Zygote
using Zygote: Pullback

function Lux.Experimental.compute_gradients(::AutoZygote, objective_function::F, data,
        ts::Lux.Experimental.TrainState) where {F}
    (loss, st, stats), back = Zygote.pullback(
        objective_function, ts.model, ts.parameters, ts.states, data)
    grads = back((one(loss), nothing, nothing))[2]
    @set! ts.states = st
    return grads, loss, stats, ts
end

end

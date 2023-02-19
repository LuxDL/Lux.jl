module LuxZygoteExt

isdefined(Base, :get_extension) ? (using Zygote) : (using ..Zygote)

using Adapt, CUDA, Lux, Setfield

Adapt.adapt_storage(::Lux.LuxCUDAAdaptor, x::Zygote.OneElement) = CUDA.cu(collect(x))

Adapt.adapt_storage(::Lux.LuxCPUAdaptor, x::Zygote.OneElement) = x

function Lux.Training.compute_gradients(::Lux.Training.ZygoteVJP,
                                        objective_function::Function, data,
                                        ts::Lux.Training.TrainState)
    (loss, st, stats), back = Zygote.pullback(ps -> objective_function(ts.model, ps,
                                                                       ts.states, data),
                                              ts.parameters)
    grads = back((one(loss), nothing, nothing))[1]
    @set! ts.states = st
    return grads, loss, stats, ts
end

end

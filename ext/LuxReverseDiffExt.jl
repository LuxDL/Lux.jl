module LuxReverseDiffExt

if isdefined(Base, :get_extension)
    using ReverseDiff
    import ReverseDiff: TrackedArray, TrackedReal, decrement_deriv!, increment_deriv!,
                        value, @grad_from_chainrules
else
    using ..ReverseDiff
    import ReverseDiff: TrackedArray, TrackedReal, decrement_deriv!, increment_deriv!,
                        value, @grad_from_chainrules
end
using Lux, Setfield

# Lux.Training
function Lux.Training.compute_gradients(::Lux.Training.ReverseDiffVJP,
                                        objective_function::Function, data,
                                        ts::Lux.Training.TrainState)
    # TODO: Implement
end

end

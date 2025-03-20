module LuxForwardDiffExt

using ADTypes: AutoForwardDiff
using ForwardDiff: gradient
using Setfield: @set!

using Lux: Lux, Utils
using Lux.Training: TrainState

include("training.jl")

end

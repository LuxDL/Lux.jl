module LuxReverseDiffExt

using ADTypes: ADTypes, AutoReverseDiff
using ArrayInterface: ArrayInterface
using Lux: Lux, LuxCPUDevice
using Lux.Experimental: TrainingBackendCache, TrainState
using ReverseDiff: ReverseDiff, TrackedArray, TrackedReal, @grad_from_chainrules

include("apply.jl")
include("utils.jl")
include("rules.jl")
include("training.jl")

end

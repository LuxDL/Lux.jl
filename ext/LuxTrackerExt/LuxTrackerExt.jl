module LuxTrackerExt

using ADTypes: AutoTracker
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore
using Lux: Lux, LuxCPUDevice
using Lux.Training: TrainingBackendCache, TrainState
using LuxCore: LuxCore, AbstractExplicitLayer
using Tracker: Tracker, TrackedArray, TrackedReal, @grad_from_chainrules

const CRC = ChainRulesCore

include("apply.jl")
include("utils.jl")
include("rules.jl")
include("training.jl")

end

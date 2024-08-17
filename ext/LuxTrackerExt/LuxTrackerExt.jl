module LuxTrackerExt

using ADTypes: AbstractADType, AutoTracker
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore
using Tracker: Tracker, TrackedArray, TrackedReal, @grad_from_chainrules

using Lux: Lux, Utils
using Lux.Training: TrainingBackendCache, TrainState
using MLDataDevices: CPUDevice

const CRC = ChainRulesCore

include("utils.jl")
include("rules.jl")
include("training.jl")

end

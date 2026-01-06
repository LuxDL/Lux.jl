module TrackerExt

using ADTypes: AbstractADType, AutoTracker
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore
using Functors: fmap
using Setfield: @set!
using Static: False, True
using Tracker: Tracker, TrackedArray, TrackedReal, @grad_from_chainrules

using Lux: Lux, Utils
using Lux.Training: Training, TrainingBackendCache, TrainState
using MLDataDevices: isleaf

const CRC = ChainRulesCore

include("utils.jl")
include("rules.jl")
include("training.jl")

end

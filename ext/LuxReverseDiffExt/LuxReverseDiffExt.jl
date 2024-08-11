module LuxReverseDiffExt

using ADTypes: ADTypes, AutoReverseDiff
using ArrayInterface: ArrayInterface
using FunctionWrappers: FunctionWrapper
using ReverseDiff: ReverseDiff, ForwardExecutor, ReverseExecutor, TrackedArray, TrackedReal,
                   @grad_from_chainrules

using Lux: Lux, Utils
using Lux.Training: TrainingBackendCache, TrainState
using LuxCore: LuxCore
using MLDataDevices: CPUDevice

include("utils.jl")
include("rules.jl")
include("training.jl")

end

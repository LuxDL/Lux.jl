module LuxReverseDiffExt

using ADTypes: ADTypes, AbstractADType, AutoReverseDiff
using ArrayInterface: ArrayInterface
using FunctionWrappers: FunctionWrapper
using Lux: Lux
using Lux.Training: TrainingBackendCache, TrainState
using LuxCore: LuxCore, AbstractExplicitLayer
using MLDataDevices: CPUDevice
using ReverseDiff: ReverseDiff, ForwardExecutor, ReverseExecutor, TrackedArray, TrackedReal,
                   @grad_from_chainrules

include("utils.jl")
include("rules.jl")
include("training.jl")

end

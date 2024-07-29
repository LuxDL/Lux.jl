module LuxReverseDiffExt

using ADTypes: ADTypes, AutoReverseDiff
using ArrayInterface: ArrayInterface
using ForwardDiff: ForwardDiff
using FunctionWrappers: FunctionWrapper
using Lux: Lux, LuxCPUDevice
using Lux.Training: TrainingBackendCache, TrainState
using LuxCore: LuxCore, AbstractExplicitLayer
using ReverseDiff: ReverseDiff, ForwardExecutor, ReverseExecutor, TrackedArray, TrackedReal,
                   @grad_from_chainrules

include("apply.jl")
include("utils.jl")
include("rules.jl")
include("training.jl")

end

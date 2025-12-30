module ReverseDiffExt

using ADTypes: ADTypes, AbstractADType, AutoReverseDiff
using ArrayInterface: ArrayInterface
using FunctionWrappers: FunctionWrapper
using Functors: fmap
using ReverseDiff:
    ReverseDiff,
    ForwardExecutor,
    ReverseExecutor,
    TrackedArray,
    TrackedReal,
    @grad_from_chainrules
using Setfield: @set!
using Static: False, True

using Lux: Lux, Utils
using Lux.Training: Training, TrainingBackendCache, TrainState
using LuxCore: LuxCore
using MLDataDevices: CPUDevice, isleaf

include("utils.jl")
include("rules.jl")
include("training.jl")

end

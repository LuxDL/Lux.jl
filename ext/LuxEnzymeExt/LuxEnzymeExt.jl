module LuxEnzymeExt

using ADTypes: AutoEnzyme
using Enzyme: Enzyme, Active, Const, Duplicated
using EnzymeCore: EnzymeCore
using Functors: fmap
using Setfield: @set!
using Static: False, True

using Lux: Lux, Utils
using Lux.Training: TrainingBackendCache, TrainState
using MLDataDevices: isleaf

include("training.jl")

end

module LuxEnzymeExt

using ADTypes: AutoEnzyme
using Enzyme: Enzyme, Active, Const, Duplicated
using EnzymeCore: EnzymeCore
using Setfield: @set!
using Static: False, True

using Lux: Lux
using Lux.Training: TrainingBackendCache, TrainState

include("training.jl")

end

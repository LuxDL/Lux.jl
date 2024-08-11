module LuxEnzymeExt

using ADTypes: AutoEnzyme
using Enzyme: Enzyme, Active, Const, Duplicated
using EnzymeCore: EnzymeCore

using Lux: Lux
using Lux.Training: TrainingBackendCache, TrainState

include("training.jl")

end

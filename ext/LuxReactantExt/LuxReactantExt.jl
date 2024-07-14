module LuxReactantExt

using Enzyme: Enzyme, Active, Const, Duplicated
using Reactant: Reactant
using Lux: Lux, ReactantBackend
using Lux.Experimental: TrainingBackendCache, TrainState
using LuxCore: LuxCore

include("utils.jl")
include("training.jl")

end

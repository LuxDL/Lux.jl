module LuxReactantExt

using Enzyme: Enzyme, Active, Const, Duplicated
using Reactant: Reactant
using Static: Static, False
using Setfield: @set!

using Lux: Lux, ReactantBackend
using Lux.Training: TrainingBackendCache, TrainState
using LuxCore: LuxCore

include("training.jl")

end

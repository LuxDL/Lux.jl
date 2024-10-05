module LuxReactantExt

using Enzyme: Enzyme, Const, Duplicated, Active
using LossFunctions: LossFunctions
using Optimisers: Optimisers
using Reactant: Reactant, @compile, TracedRArray
using Setfield: @set!
using Static: False
using Statistics: mean

using Lux: Lux, LuxOps, Training, LossFunctionImpl
using Lux.Training: TrainingBackendCache, ReactantBackend

include("overrides.jl")
include("training.jl")

end

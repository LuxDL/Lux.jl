module LuxReactantExt

using Enzyme: Enzyme, Const, Duplicated, Active
using Optimisers: Optimisers
using Reactant: Reactant, @compile
using Setfield: @set!
using Static: False

using Lux: Lux, Training
using Lux.Training: TrainingBackendCache, ReactantBackend

include("training.jl")

end

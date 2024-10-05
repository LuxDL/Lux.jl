module LuxReactantExt

using Enzyme: Enzyme, Const, Duplicated, Active
using Optimisers: Optimisers
using Reactant: Reactant, @compile, TracedRArray
using Setfield: @set!
using Static: False

using Lux: Lux, LuxOps, Training
using Lux.Training: TrainingBackendCache, ReactantBackend

include("overrides.jl")
include("training.jl")

end

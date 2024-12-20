module LuxReactantExt

using Enzyme: Enzyme, Const, Duplicated, Active
using Optimisers: Optimisers
using Reactant: Reactant, @compile, AnyTracedRArray, TracedRArray, TracedRNumber
using Setfield: @set!
using Static: False

using Lux: Lux, LuxOps, Training, Utils
using Lux.Training: TrainingBackendCache, ReactantBackend

include("patches.jl")
include("training.jl")

end

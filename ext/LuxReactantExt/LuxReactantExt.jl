module LuxReactantExt

using Adapt: adapt
using ArgCheck: @argcheck
using ConcreteStructs: @concrete
using Enzyme: Enzyme, Active, Const, Duplicated
using Functors: fmapstructure, fmap
using Optimisers: Optimisers, Descent, Leaf
using Random: AbstractRNG, Xoshiro
using Reactant: Reactant
using Lux: Lux, LuxEltypeAdaptor, ReactantBackend
using Lux.Experimental: TrainingBackendCache, TrainState
using LuxCore: LuxCore, AbstractExplicitLayer

include("utils.jl")
include("train.jl")
include("optimizers.jl")

end

module LuxReactantExt

using Adapt: adapt
using ArgCheck: @argcheck
using ConcreteStructs: @concrete
using Enzyme: Enzyme, Active, Const, Duplicated
using Functors: fmapstructure, fmap
using Random: AbstractRNG, Xoshiro
using Reactant: Reactant
using Lux: Lux, LuxEltypeAdaptor, AutoReactant
using LuxCore: LuxCore, AbstractExplicitLayer

include("utils.jl")

# compile just the model. This allows us to run part of the model in vanilla LLVM. Needed
# for cases where we can't currently compile via Reactant or where XLA is not great
# for the model.
include("layer.jl")

# compile the entire training loop
include("train.jl")

end

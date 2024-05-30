module LuxReactantExt

using Adapt: adapt
using ArgCheck: @argcheck
using Enzyme: Enzyme
using Functors: fmapstructure, fmap
using Markdown: @md_str
using Random: AbstractRNG, Xoshiro
using Reactant: Reactant
using Lux: Lux, LuxEltypeAdaptor
using LuxCore: LuxCore, AbstractExplicitLayer

# compile just the model. This allows us to run part of the model in vanilla LLVM. Needed
# for cases where we can't currently compile via Reactant or where XLA is not great
# for the model.
include("layer.jl")

include("train.jl")

end

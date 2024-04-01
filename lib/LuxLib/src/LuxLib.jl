module LuxLib

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ChainRulesCore: ChainRulesCore
    using FastClosures: @closure
    using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel
    using LuxCore: LuxCore
    using Markdown: @doc_str
    using NNlib: NNlib
    using Random: Random, AbstractRNG, rand!
    using Reexport: @reexport
    using Statistics: Statistics, mean, var, varm
end

@reexport using NNlib

const CRC = ChainRulesCore
const KA = KernelAbstractions

include("utils.jl")

# Low-Level Implementations
include("impl/groupnorm.jl")
include("impl/normalization.jl")

# User Facing
include("api/batchnorm.jl")
include("api/dropout.jl")
include("api/groupnorm.jl")
include("api/instancenorm.jl")
include("api/layernorm.jl")

export batchnorm, groupnorm, instancenorm, layernorm, alpha_dropout, dropout

end

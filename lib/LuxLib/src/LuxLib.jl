module LuxLib

import PrecompileTools

PrecompileTools.@recompile_invalidations begin
    using ChainRulesCore, KernelAbstractions, Markdown, NNlib, Random, Reexport, Statistics
end

@reexport using NNlib
import ChainRulesCore as CRC
import KernelAbstractions as KA

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

export batchnorm, groupnorm, instancenorm, layernorm
export alpha_dropout, dropout

end

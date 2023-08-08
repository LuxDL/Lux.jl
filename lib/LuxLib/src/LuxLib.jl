module LuxLib

using Reexport

using ChainRulesCore, Markdown, Random, Statistics
import ChainRulesCore as CRC

@reexport using NNlib

using KernelAbstractions
import KernelAbstractions as KA

# Extensions
using PackageExtensionCompat
function __init__()
    @require_extensions
end

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

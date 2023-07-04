module LuxLib

using Reexport

using ChainRulesCore, Markdown, Random, Statistics
import ChainRulesCore as CRC

@reexport using NNlib

using KernelAbstractions
import KernelAbstractions as KA

# Extensions
#=
using PackageExtensionCompat
function __init__()
    @require_extensions
end
=#
if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        # Handling AD Packages
        ## Handling ForwardDiff
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/LuxLibForwardDiffExt.jl")
        end
        ## Handling Tracker
        @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
            include("../ext/LuxLibTrackerExt.jl")
        end
        ## Handling ReverseDiff
        @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
            include("../ext/LuxLibReverseDiffExt.jl")
        end

        # Accelerator Support
        ## Handling CUDA
        @require LuxCUDA="d0bbae9a-e099-4d5b-a835-1c6931763bda" begin
            include("../ext/LuxLibLuxCUDAExt.jl")

            @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
                include("../ext/LuxLibLuxCUDATrackerExt.jl")
            end
        end
    end
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

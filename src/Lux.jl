module Lux

# Accelerator Support
using CUDA, cuDNN
# Neural Network Backend
using NNlib
import LuxLib  ## In v0.5 we can starting `using`. For v0.4, there will be naming conflicts
# Julia StdLibs
using LinearAlgebra, Markdown, Random, SparseArrays, Statistics
# Parameter Manipulation
using Functors, Setfield
import Adapt: adapt, adapt_storage
# Automatic Differentiation
using ChainRulesCore
import ChainRulesCore as CRC
# Smaller Stacktraces -- Till we have better solution in Base
import TruncatedStacktraces
import TruncatedStacktraces: @truncate_stacktrace

# LuxCore
using LuxCore
import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer, initialparameters,
                initialstates, parameterlength, statelength, update_state, trainmode,
                testmode, setup, apply

const use_cuda = Ref{Union{Nothing, Bool}}(nothing)

# Utilities
include("utils.jl")
# Data Transfer Utilities
include("adapt.jl")
# Layer Implementations
include("layers/basic.jl")
include("layers/containers.jl")
include("layers/normalize.jl")
include("layers/conv.jl")
include("layers/dropout.jl")
include("layers/recurrent.jl")
# Pretty Printing
include("layers/display.jl")
include("stacktraces.jl")
# AutoDiff
include("chainrules.jl")

# Experimental
include("contrib/map.jl")
include("contrib/training.jl")
include("contrib/freeze.jl")
include("contrib/share_parameters.jl")

# Deprecations
include("deprecated.jl")

# Extensions
if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        # Handling ComponentArrays
        @require ComponentArrays="b0b7db55-cfe3-40fc-9ded-d10e2dbeff66" begin
            include("../ext/LuxComponentArraysExt.jl")
            # These definitely needs to be upstreamed
            @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin include("../ext/LuxComponentArraysTrackerExt.jl") end
            @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin include("../ext/LuxComponentArraysZygoteExt.jl") end
            @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" begin include("../ext/LuxComponentArraysReverseDiffExt.jl") end
        end

        # Flux InterOp
        @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin include("../ext/LuxFluxTransformExt.jl") end

        # FillArrays
        @require FillArrays="1a297f60-69ca-5386-bcde-b61e274b549b" begin include("../ext/LuxFillArraysExt.jl") end

        # Automatic Differentiation
        ## Zygote InterOp
        @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin include("../ext/LuxZygoteExt.jl") end
        ## Tracker InterOp
        @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin include("../ext/LuxTrackerExt.jl") end
    end
end

# Data Transfer
export cpu, gpu
# Layers
export Chain, Parallel, SkipConnection, PairwiseFusion, BranchLayer, Maxout
export Bilinear, Dense, Embedding, Scale
export Conv, ConvTranspose, CrossCor, MaxPool, MeanPool, GlobalMaxPool, GlobalMeanPool,
       AdaptiveMaxPool, AdaptiveMeanPool, Upsample, PixelShuffle
export AlphaDropout, Dropout, VariationalHiddenDropout
export BatchNorm, GroupNorm, InstanceNorm, LayerNorm
export WeightNorm
export NoOpLayer, ReshapeLayer, SelectDim, FlattenLayer, WrappedFunction, ActivationFunction
export RNNCell, LSTMCell, GRUCell, Recurrence, StatefulRecurrentCell
export SamePad

# Extension Exports: Flux
function transform end

struct FluxLayer{L, RE, I} <: Lux.AbstractExplicitLayer
    layer::L
    re::RE
    init_parameters::I
end

export transform, FluxLayer

end

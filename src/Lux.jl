module Lux

# Accelerator Support
using CUDA
using cuDNN
# Neural Network Backend
using NNlib
import LuxLib  ## In v0.5 we can starting `using`. For v0.4, there will be naming conflicts
# Julia StdLibs
using Random, Statistics, LinearAlgebra, SparseArrays
# Parameter Manipulation
using Functors, Setfield
import Adapt: adapt, adapt_storage
# Arrays
using FillArrays
# Automatic Differentiation
using ChainRulesCore, Zygote
import ChainRulesCore as CRC
# Docstrings
using Markdown

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
# Neural Network Backend
include("nnlib.jl")
# Pretty Printing
include("layers/display.jl")
# AutoDiff
include("autodiff.jl")

# Extensions
if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        # Handling ComponentArrays
        @require ComponentArrays="b0b7db55-cfe3-40fc-9ded-d10e2dbeff66" begin include("../ext/LuxComponentArraysExt.jl") end

        # Flux InterOp
        @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin include("../ext/LuxFluxTransformExt.jl") end
    end
end

# Experimental
include("contrib/map.jl")
include("contrib/training.jl")
include("contrib/freeze.jl")
include("contrib/share_parameters.jl")

# Deprecations
include("deprecated.jl")

# Snoop Precompile
import SnoopPrecompile
import Preferences

SnoopPrecompile.@precompile_all_calls begin include("precompile.jl") end

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

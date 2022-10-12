module Lux

# Accelerator Support
using CUDA
using CUDA.CUDNN
# Neural Network Backend
using NNlib
import LuxLib  ## In v0.5 we can starting `using`. For v0.4, there will be naming conflicts
# Julia StdLibs
using Random, Statistics, LinearAlgebra, SparseArrays
# Parameter Manipulation
using Functors, Setfield
import Adapt: adapt, adapt_storage
# Arrays
using FillArrays, ComponentArrays
# Automatic Differentiation
using ChainRulesCore, Zygote
# Optional Dependency
using Requires
# Docstrings
using Markdown
# Optimisers + ComponentArrays
using Optimisers

const use_cuda = Ref{Union{Nothing, Bool}}(nothing)

# Utilities
include("utils.jl")
# Core
include("core.jl")
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
# Flux to Lux
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("transform.jl")
end

# Experimental
include("contrib/map.jl")
include("contrib/training.jl")
include("contrib/freeze.jl")

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
export Dropout, VariationalHiddenDropout
export BatchNorm, GroupNorm, LayerNorm
export WeightNorm
export NoOpLayer, ReshapeLayer, SelectDim, FlattenLayer, WrappedFunction, ActivationFunction
export RNNCell, LSTMCell, GRUCell, Recurrence, StatefulRecurrentCell
export SamePad

end

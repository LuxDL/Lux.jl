module Lux

# Some core imports
using Preferences, Reexport
# Neural Network Backend
@reexport using LuxLib
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
@reexport using LuxCore
import LuxCore: AbstractExplicitLayer,
    AbstractExplicitContainerLayer,
    initialparameters,
    initialstates,
    parameterlength,
    statelength,
    update_state,
    trainmode,
    testmode,
    setup,
    apply,
    display_name

# Standard Weight Initializations
using WeightInitializers
import WeightInitializers: randn32,
    rand32, ones32, zeros32, glorot_uniform, glorot_normal, kaiming_normal, kaiming_uniform
import WeightInitializers: _nfan

const use_cuda = Ref{Union{Nothing, Bool}}(nothing)
const ACCELERATOR_STATE_CHANGED = Ref{Bool}(false)

const NAME_TYPE = Union{Nothing, String, Symbol}

# Utilities
include("utils.jl")
# Data Transfer Utilities
include("device.jl")
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
using PackageExtensionCompat
function __init__()
    @require_extensions
end

# Data Transfer
export cpu, gpu, cpu_device, gpu_device, LuxCPUDevice, LuxCUDADevice, LuxAMDGPUDevice
# Layers
export Chain, Parallel, SkipConnection, PairwiseFusion, BranchLayer, Maxout
export Bilinear, Dense, Embedding, Scale
export Conv,
    ConvTranspose,
    CrossCor,
    MaxPool,
    MeanPool,
    GlobalMaxPool,
    GlobalMeanPool,
    AdaptiveMaxPool,
    AdaptiveMeanPool,
    Upsample,
    PixelShuffle
export AlphaDropout, Dropout, VariationalHiddenDropout
export BatchNorm, GroupNorm, InstanceNorm, LayerNorm
export WeightNorm
export NoOpLayer, ReshapeLayer, SelectDim, FlattenLayer, WrappedFunction
export RNNCell, LSTMCell, GRUCell, Recurrence, StatefulRecurrentCell
export SamePad

# Extension Exports: Flux
function transform end

struct FluxLayer{L, RE, I} <: AbstractExplicitLayer
    layer::L
    re::RE
    init_parameters::I
end

export transform, FluxLayer

end

module Lux

# Some core imports
using Preferences, Reexport
# Neural Network Backend
@reexport using LuxLib
# Julia StdLibs
using LinearAlgebra, Markdown, Random, SparseArrays, Statistics
# Parameter Manipulation
using Functors, Setfield
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

# Device Management
@reexport using LuxDeviceUtils, WeightInitializers
import LuxDeviceUtils: AbstractLuxDevice, AbstractLuxGPUDevice, AbstractLuxDeviceAdaptor

const NAME_TYPE = Union{Nothing, String, Symbol}

# Utilities
include("utils.jl")
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
include("extensions.jl")
using PackageExtensionCompat
function __init__()
    @require_extensions
end

# Layers
export cpu, gpu
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

export transform, FluxLayer

end

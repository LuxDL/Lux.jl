module ExplicitFluxLayers

using Statistics, NNlib, CUDA, Random, Setfield, ChainRulesCore
import NNlibCUDA: batchnorm
import Flux
import Flux:
    zeros32,
    ones32,
    glorot_normal,
    glorot_uniform,
    convfilter,
    expand,
    calc_padding,
    DenseConvDims,
    _maybetuple_string,
    reshape_cell_output

# Core
include("core.jl")

# Utilities
include("utils.jl")

# Layer Implementations
include("layers/basic.jl")
include("layers/normalize.jl")
include("layers/conv.jl")

# Transition to Explicit Layers
include("transform.jl")

# Pretty Printing
include("layers/display.jl")

# AutoDiff
include("autodiff.jl")

end

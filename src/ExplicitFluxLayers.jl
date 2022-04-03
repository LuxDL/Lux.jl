module ExplicitFluxLayers

using Statistics, NNlib, CUDA, Random, Setfield, ChainRulesCore, Octavian, LinearAlgebra, FillArrays
import NNlibCUDA: batchnorm, cudnnBNForward!
using Flux: Flux
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

# Neural Network Backend
include("nnlib.jl")

# Layer Implementations
include("layers/basic.jl")
include("layers/normalize.jl")
include("layers/conv.jl")

# Transition to Explicit Layers
include("transform.jl")

# Pretty Printing
include("layers/display.jl")

# Sparse Layers
include("sparse.jl")

# AutoDiff
include("autodiff.jl")

end

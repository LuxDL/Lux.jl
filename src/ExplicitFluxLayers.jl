module ExplicitFluxLayers

const EFL = ExplicitFluxLayers

using Statistics,
    NNlib,
    CUDA,
    Random,
    Setfield,
    ChainRulesCore,
    Octavian,
    LinearAlgebra,
    FillArrays,
    Functors,
    ComponentArrays,
    Zygote
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
    reshape_cell_output,
    _dropout_mask,
    gpu,
    cpu

# Core
include("core.jl")

# Utilities
include("utils.jl")

# Layer Implementations
include("layers/basic.jl")
include("layers/normalize.jl")
include("layers/conv.jl")
include("layers/dropout.jl")

# Neural Network Backend
include("nnlib.jl")

# Transition to Explicit Layers
include("transform.jl")

# Pretty Printing
include("layers/display.jl")

# Sparse Layers
include("sparse.jl")

# AutoDiff
include("autodiff.jl")

export EFL

end

module ExplicitFluxLayers

const EFL = ExplicitFluxLayers

# Accelerator Support
using CUDA
# Neural Network Backend
using NNlib
import NNlibCUDA: batchnorm
# Julia StdLibs
using Random, Statistics, LinearAlgebra, SparseArrays
# Parameter Manipulation
using Functors, Setfield
import Adapt: adapt, adapt_storage
# Arrays
using FillArrays, ComponentArrays
# Automatic Differentiation
using ChainRulesCore, Zygote
# Optimization
using Optimisers
# Optional Dependency
using Requires

const use_cuda = Ref{Union{Nothing,Bool}}(nothing)

# Data Transfer Utilities
include("adapt.jl")
# Utilities
include("utils.jl")
# Core
include("core.jl")
# Layer Implementations
include("layers/basic.jl")
include("layers/normalize.jl")
include("layers/conv.jl")
include("layers/dropout.jl")
# Neural Network Backend
include("nnlib.jl")
# Pretty Printing
include("layers/display.jl")
# AutoDiff
include("autodiff.jl")
# Transition to Explicit Layers
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("transform.jl")
end


export EFL

end

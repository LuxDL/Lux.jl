module Boltz

using CUDA, Lux, NNlib, Random, Statistics

# Loading Pretained Weights
using Artifacts, JLD2, LazyArtifacts

# TODO(@avik-pal): We want to have generic Lux implementaions for Metalhead models
# We can automatically convert several Metalhead.jl models to Lux
using Flux2Lux, Metalhead

# Mark certain parts of layers as non-differentiable
import ChainRulesCore

# Utility Functions
include("utils.jl")

# Vision Models
## Vision Transformer Implementation
include("vision/vit.jl")
## VGG Nets
include("vision/vgg.jl")
## Metalhead to Lux
include("vision/metalhead.jl")

# Exports
export alexnet, convmixer, densenet, googlenet, mobilenet, resnet, resnext, vgg,
       vision_transformer

end

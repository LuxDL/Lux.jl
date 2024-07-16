module LuxLib

using ArrayInterface: ArrayInterface, fast_scalar_indexing
using ChainRulesCore: ChainRulesCore, NoTangent, HasReverseMode, RuleConfig
using DispatchDoctor: @stable
using EnzymeCore: EnzymeCore, EnzymeRules
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra, BLAS, mul!
using LuxCore: LuxCore
using LuxDeviceUtils: get_device_type, LuxCUDADevice, LuxCPUDevice, AbstractLuxGPUDevice,
                      AbstractLuxDevice
using Markdown: @doc_str
using NNlib: NNlib, ConvDims, conv, conv!, relu, sigmoid_fast, swish, σ, ∇conv_data,
             ∇conv_filter
using Random: Random, AbstractRNG, rand!
using Reexport: @reexport
using Statistics: Statistics, mean, var
using UnrolledUtilities: unrolled_any, unrolled_all

@reexport using NNlib

const CRC = ChainRulesCore

include("utils.jl")

# User Facing
include("api/bias_activation.jl")
include("api/batchnorm.jl")
include("api/broadcast.jl")
include("api/dropout.jl")
include("api/groupnorm.jl")
include("api/instancenorm.jl")
include("api/layernorm.jl")
include("api/dense.jl")
include("api/conv.jl")

# Low-Level Implementations
include("impl/bias_activation.jl")
include("impl/broadcast.jl")
include("impl/dropout.jl")
include("impl/fused_dense.jl")
include("impl/fused_conv.jl")
include("impl/forward_diff.jl")
include("impl/normalization.jl")

include("deprecations.jl")

export batchnorm, groupnorm, instancenorm, layernorm, alpha_dropout, dropout
export fused_dense_bias_activation, fused_conv_bias_activation
export fast_activation!!

end

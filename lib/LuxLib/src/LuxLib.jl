module LuxLib

using ArrayInterface: ArrayInterface, fast_scalar_indexing, can_setindex
using ChainRulesCore: ChainRulesCore, NoTangent, HasReverseMode, RuleConfig
using DispatchDoctor: @stable
using EnzymeCore: EnzymeCore, EnzymeRules
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using KernelAbstractions: KernelAbstractions, @kernel, @Const, @index
using LinearAlgebra: LinearAlgebra, BLAS, mul!
using LoopVectorization: indices, @tturbo
using LuxCore: LuxCore
using Markdown: @doc_str
using MLDataDevices: get_device_type, AMDGPUDevice, CUDADevice, CPUDevice,
                     AbstractGPUDevice, AbstractDevice
using NNlib: NNlib, ConvDims, conv, conv!, relu, gelu, σ, ∇conv_data, ∇conv_filter
using Random: Random, AbstractRNG, rand!
using Reexport: @reexport
using StaticArraysCore: StaticArraysCore, StaticArray, StaticVector
using Statistics: Statistics, mean, var
using SLEEFPirates: SLEEFPirates
using UnrolledUtilities: unrolled_any, unrolled_all, unrolled_filter, unrolled_mapreduce
using VectorizedStatistics: vmean, vvar

@reexport using NNlib

const CRC = ChainRulesCore
const KA = KernelAbstractions

include("utils.jl")
include("patches.jl")

# User Facing
include("api/activation.jl")
include("api/bias_activation.jl")
include("api/batchnorm.jl")
include("api/dropout.jl")
include("api/groupnorm.jl")
include("api/instancenorm.jl")
include("api/layernorm.jl")
include("api/dense.jl")
include("api/conv.jl")

# Low-Level Implementations
include("impl/activation.jl")
include("impl/affine_normalize.jl")
include("impl/bias_activation.jl")
include("impl/dropout.jl")
include("impl/fast_ops.jl")
include("impl/fused_dense.jl")
include("impl/fused_conv.jl")
include("impl/forward_diff.jl")
include("impl/matmul.jl")
include("impl/normalization.jl")

include("deprecations.jl")

export batchnorm, groupnorm, instancenorm, layernorm, alpha_dropout, dropout
export fused_dense_bias_activation, fused_conv_bias_activation
export fast_activation!!
export bias_activation, bias_activation!!

end

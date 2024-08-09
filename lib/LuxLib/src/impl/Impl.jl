module Impl

using ArrayInterface: ArrayInterface, aos_to_soa
using DispatchDoctor: @stable
using FastClosures: @closure
using StaticArraysCore: StaticVector, SArray
using Static: StaticBool, True, False, static
using UnrolledUtilities: unrolled_mapreduce

using ChainRulesCore: ChainRulesCore, NoTangent, HasReverseMode, RuleConfig
using EnzymeCore: EnzymeCore, EnzymeRules
using ForwardDiff: ForwardDiff

using KernelAbstractions: KernelAbstractions, @kernel, @Const, @index

using LoopVectorization: LoopVectorization, @turbo, @tturbo, indices
using Octavian: Octavian
using Polyester: @batch

using LinearAlgebra: LinearAlgebra, mul!
using Random: Random, AbstractRNG, rand!
using Statistics: Statistics, mean, var

using LuxCore: LuxCore
using MLDataDevices: get_device_type, AMDGPUDevice, CUDADevice, AbstractGPUDevice,
                     AbstractDevice
using NNlib: NNlib, ConvDims

using ..LuxLib: Optional, Numeric, ∂∅, internal_operation_mode, AbstractInternalArrayOpMode,
                GenericBroadcastOp, GPUBroadcastOp, LoopedArrayOp, Utils, Traits, System,
                get_utils

const CRC = ChainRulesCore
const KA = KernelAbstractions
const LV = LoopVectorization

include("activation.jl")
include("batched_mul.jl")
include("batchnorm.jl")
include("bias_activation.jl")
include("common_ops.jl")
include("conv.jl")
include("dense.jl")
include("dropout.jl")
include("forward_diff.jl")
include("groupnorm.jl")
include("matmul.jl")
include("normalization.jl")

end

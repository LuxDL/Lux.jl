module Impl

using DispatchDoctor: @stable
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra, mul!
using LuxCore: LuxCore
using MLDataDevices: get_device_type, AMDGPUDevice, CUDADevice, CPUDevice,
                     AbstractGPUDevice, AbstractDevice
using NNlib: NNlib, ConvDims
using Random: Random, AbstractRNG, rand!
using Static: StaticBool, True, False
using StaticArraysCore: StaticVector, SArray
using UnrolledUtilities: unrolled_mapreduce

using KernelAbstractions: KernelAbstractions

using LoopVectorization: LoopVectorization, @turbo, @tturbo, indices
using Octavian: Octavian
using Polyester: @batch

using ChainRulesCore: ChainRulesCore, NoTangent, HasReverseMode, RuleConfig
using EnzymeCore: EnzymeCore, EnzymeRules

using ..LuxLib: Numeric, Optional, internal_operation_mode, AbstractInternalArrayOpMode,
                GenericBroadcastOp, GPUBroadcastOp, LoopedArrayOp
using ..Utils
using ..System
using ..Traits

const CRC = ChainRulesCore
const KA = KernelAbstractions
const LV = LoopVectorization

const ∂∅ = NoTangent()

include("activation.jl")
include("batched_mul.jl")
include("bias_activation.jl")
include("common_ops.jl")
include("conv.jl")
include("dense.jl")
include("dropout.jl")
include("matmul.jl")

end

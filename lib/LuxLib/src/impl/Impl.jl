module Impl

using ArrayInterface: ArrayInterface, aos_to_soa
using DispatchDoctor: @stable
using FastClosures: @closure
using StaticArraysCore: StaticVector, SArray
using Static: StaticBool, True, False, static

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
using MLDataDevices: get_device_type, CPUDevice, AMDGPUDevice, CUDADevice, XLADevice,
                     AbstractGPUDevice, AbstractDevice
using NNlib: NNlib, ConvDims

using ..LuxLib: Optional, Numeric, ∂∅, internal_operation_mode, AbstractInternalArrayOpMode,
                GenericBroadcastOp, GPUBroadcastOp, LoopedArrayOp
using ..Utils: Utils, NotaNumber, batchview, concrete_bias_act_output_eltype, contiguous,
               copy_drop_gradients, eltype_mismatch, expand_batchdim,
               maybe_reduce_BLAS_threads, ofeltype_array, only_derivative, remove_tracking,
               reset_BLAS_threads, run_ka_kernel, safe_eltype, safe_vec, safe_warning,
               unsafe_known, unrolled_mapreduce, @enzyme_alternative
using ..Traits: activation_intermediate_not_needed, activation_has_rrule, is_mutable_array,
                fuse_cpu_activation
using ..System: explicit_blas_loaded, use_octavian, fits_in_l1cache, fits_in_l2cache,
                fits_in_l3cache

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
include("layernorm.jl")
include("matmul.jl")
include("normalization.jl")

end

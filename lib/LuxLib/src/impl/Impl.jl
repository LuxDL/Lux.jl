module Impl

using DispatchDoctor: @stable
using FastClosures: @closure
using Static: True, False
using UnrolledUtilities: unrolled_mapreduce

using KernelAbstractions: KernelAbstractions

using LoopVectorization: LoopVectorization, @turbo, @tturbo, indices
using Polyester: @batch

using ChainRulesCore: ChainRulesCore, NoTangent, HasReverseMode, RuleConfig
using EnzymeCore: EnzymeCore, EnzymeRules

using ..LuxLib: Numeric, internal_operation_mode, AbstractInternalArrayOpMode,
                GenericBroadcastOp, GPUBroadcastOp, LoopedArrayOp
using ..Utils
using ..Traits

const CRC = ChainRulesCore
const KA = KernelAbstractions
const LV = LoopVectorization

const ∂∅ = NoTangent()

include("activation.jl")

end

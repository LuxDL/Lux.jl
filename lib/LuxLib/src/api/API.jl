module API

using ChainRulesCore: ChainRulesCore
using NNlib: NNlib, ConvDims
using Random: Random, AbstractRNG
using Static: Static, StaticBool, True, False

using ..LuxLib: Optional
using ..Impl
using ..Utils

const CRC = ChainRulesCore

include("activation.jl")
include("batched_mul.jl")
include("bias_activation.jl")
include("conv.jl")
include("dense.jl")
include("dropout.jl")

export alpha_dropout, dropout
export bias_activation, bias_activation!!
export batched_matmul
export fast_activation, fast_activation!!
export fused_conv_bias_activation
export fused_dense_bias_activation

end

@reexport using .API

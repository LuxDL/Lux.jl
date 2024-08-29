module API

using ChainRulesCore: ChainRulesCore
using Markdown: @doc_str
using NNlib: NNlib, ConvDims
using Random: Random, AbstractRNG
using Static: Static, StaticBool, static

using ..LuxLib: Optional
using ..Impl: Impl, select_fastest_activation
using ..Utils: default_epsilon, expand_batchdim, remove_tracking

const CRC = ChainRulesCore

# The names are aliased so we define constants for them
for op in (:batched_matmul, :batchnorm, :bias_activation, :bias_activation!!,
    :dropout, :alpha_dropout, :groupnorm, :instancenorm, :layernorm,
    :activation, :activation!!, :fused_conv, :fused_dense)
    impl_op = Symbol(op, :_impl)
    @eval const $impl_op = Impl.$op
end

include("activation.jl")
include("batched_mul.jl")
include("batchnorm.jl")
include("bias_activation.jl")
include("conv.jl")
include("dense.jl")
include("dropout.jl")
include("groupnorm.jl")
include("instancenorm.jl")
include("layernorm.jl")

export alpha_dropout, dropout
export bias_activation, bias_activation!!
export batched_matmul
export batchnorm, groupnorm, instancenorm, layernorm
export fast_activation, fast_activation!!
export fused_conv_bias_activation
export fused_dense_bias_activation

end

@reexport using .API

module LuxLib

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArrayInterface: ArrayInterface
    using ChainRulesCore: ChainRulesCore
    using FastClosures: @closure
    using GPUArraysCore: AnyGPUArray
    using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel
    using LinearAlgebra: LinearAlgebra, mul!
    using LuxCore: LuxCore
    using Markdown: @doc_str
    using NNlib: NNlib
    using Random: Random, AbstractRNG, rand!
    using Reexport: @reexport
    using Statistics: Statistics, mean, std, var
end

@reexport using NNlib

const CRC = ChainRulesCore
const KA = KernelAbstractions

include("utils.jl")

# Low-Level Implementations
include("impl/groupnorm.jl")
include("impl/normalization.jl")
include("impl/fused_dense.jl")
include("impl/fused_conv.jl")

# User Facing
include("api/batchnorm.jl")
include("api/dropout.jl")
include("api/groupnorm.jl")
include("api/instancenorm.jl")
include("api/layernorm.jl")
include("api/dense.jl")
include("api/conv.jl")

export batchnorm, groupnorm, instancenorm, layernorm, alpha_dropout, dropout
export fused_dense_bias_activation, fused_conv_bias_activation

end

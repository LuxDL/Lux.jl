module API

using ChainRulesCore: ChainRulesCore
using Random: Random, AbstractRNG
using Static: Static, StaticBool, True, False

using ..LuxLib: Optional
using ..Impl
using ..Utils

const CRC = ChainRulesCore

include("activation.jl")
include("batched_mul.jl")
include("bias_activation.jl")
include("dropout.jl")

export alpha_dropout, dropout
export bias_activation, bias_activation!!
export batched_matmul
export fast_activation, fast_activation!!

end

@reexport using .API

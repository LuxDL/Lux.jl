module API

using Random: Random, AbstractRNG
using Static: Static, StaticBool, True, False

using ..Impl
using ..Utils

include("activation.jl")
include("batched_mul.jl")
include("dropout.jl")

export alpha_dropout, dropout
export batched_matmul
export fast_activation, fast_activation!!

end

using .API

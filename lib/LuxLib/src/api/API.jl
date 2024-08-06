module API

using ..Impl
using ..Utils

include("activation.jl")
include("batched_mul.jl")

export batched_matmul
export fast_activation, fast_activation!!

end

using .API

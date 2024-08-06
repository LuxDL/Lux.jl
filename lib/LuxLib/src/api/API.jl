module API

using ..Impl

include("activation.jl")

export fast_activation, fast_activation!!

end

using .API

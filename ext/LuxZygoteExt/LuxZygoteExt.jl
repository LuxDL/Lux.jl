module LuxZygoteExt

using ArgCheck: @argcheck
using ADTypes: AutoZygote
using ChainRulesCore: ChainRulesCore
using Lux: Lux
using Setfield: @set!
using Zygote: Zygote

const CRC = ChainRulesCore

Lux._is_extension_loaded(::Val{:Zygote}) = true

include("training.jl")
include("batched_ad.jl")
include("nested_ad.jl")

end

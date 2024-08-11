module LuxZygoteExt

using ArgCheck: @argcheck
using ADTypes: AutoZygote
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using Setfield: @set!
using Zygote: Zygote

using Lux: Lux
using MLDataDevices: get_device_type, CPUDevice

const CRC = ChainRulesCore

Lux.is_extension_loaded(::Val{:Zygote}) = true

include("training.jl")

include("batched_autodiff.jl")
include("nested_autodiff.jl")

end

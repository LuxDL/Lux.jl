module LuxZygoteExt

using ArgCheck: @argcheck
using ADTypes: AutoZygote
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using Lux: Lux
using LuxDeviceUtils: get_device_type, LuxCPUDevice
using Setfield: @set!
using Zygote: Zygote

const CRC = ChainRulesCore

@inline Lux._is_extension_loaded(::Val{:Zygote}) = true

@inline Lux.__zygote_rule_config() = Zygote.ZygoteRuleConfig()

include("training.jl")
include("batched_ad.jl")
include("nested_ad.jl")

end

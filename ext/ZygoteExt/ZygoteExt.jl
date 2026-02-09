module ZygoteExt

using ADTypes: AutoZygote
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using Setfield: @set!
using Zygote: Zygote

using Lux: Lux
using MLDataDevices: get_device_type, CPUDevice

const CRC = ChainRulesCore

Lux.is_extension_loaded(::Val{:Zygote}) = true

Lux.AutoDiffInternalImpl.rule_config(::Val{:Zygote}) = Zygote.ZygoteRuleConfig()

include("training.jl")

include("batched_autodiff.jl")
include("nested_autodiff.jl")

end

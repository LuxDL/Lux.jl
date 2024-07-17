module LuxCoreChainRulesCoreExt

using ChainRulesCore: @non_differentiable
using LuxCore: LuxCore
using Random: AbstractRNG

@non_differentiable LuxCore.replicate(::AbstractRNG)

end

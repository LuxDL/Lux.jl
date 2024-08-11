module LuxCoreChainRulesCoreExt

using ChainRulesCore: ChainRulesCore, @non_differentiable
using LuxCore: LuxCore, AbstractExplicitLayer
using Random: AbstractRNG

@non_differentiable LuxCore.replicate(::AbstractRNG)

function ChainRulesCore.rrule(::typeof(getproperty), m::AbstractExplicitLayer, x::Symbol)
    mₓ = getproperty(m, x)
    ∇getproperty(_) = ntuple(Returns(ChainRulesCore.NoTangent()), 3)
    return mₓ, ∇getproperty
end

end

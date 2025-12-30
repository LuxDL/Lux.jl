module ChainRulesCoreExt

using ChainRulesCore: ChainRulesCore, NoTangent, @non_differentiable
using LuxCore: LuxCore, AbstractLuxLayer
using Random: AbstractRNG

@non_differentiable LuxCore.replicate(::AbstractRNG)

function ChainRulesCore.rrule(::typeof(getproperty), m::AbstractLuxLayer, x::Symbol)
    mₓ = getproperty(m, x)
    ∇getproperty(_) = ntuple(Returns(NoTangent()), 3)
    return mₓ, ∇getproperty
end

# StatefulLuxLayer
@non_differentiable LuxCore.StatefulLuxLayerImpl.get_state(::Any)
@non_differentiable LuxCore.StatefulLuxLayerImpl.set_state!(::Any...)

function ChainRulesCore.rrule(
    ::typeof(LuxCore.StatefulLuxLayerImpl.StatefulLuxLayer),
    model::AbstractLuxLayer,
    ps,
    st,
    st_any,
    fixed_state_type,
)
    slayer = LuxCore.StatefulLuxLayer(model, ps, st, st_any, fixed_state_type)
    function ∇StatefulLuxLayer(Δ)
        return NoTangent(), NoTangent(), Δ.ps, NoTangent(), NoTangent(), NoTangent()
    end
    return slayer, ∇StatefulLuxLayer
end

function ChainRulesCore.rrule(
    ::typeof(getproperty), s::LuxCore.StatefulLuxLayerImpl.StatefulLuxLayer, name::Symbol
)
    y = getproperty(s, name)
    ∇getproperty = let s = s, name = name
        (Δ) -> begin
            name === :ps && return NoTangent(),
            ChainRulesCore.Tangent{typeof(s)}(; ps=Δ),
            NoTangent()
            return NoTangent(), NoTangent(), NoTangent()
        end
    end
    return y, ∇getproperty
end

end

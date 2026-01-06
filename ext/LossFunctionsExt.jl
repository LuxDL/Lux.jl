module LossFunctionsExt

using ArrayInterface: fast_scalar_indexing
using ChainRulesCore: ChainRulesCore, NoTangent, @thunk
using EnzymeCore: EnzymeCore, EnzymeRules
using FastClosures: @closure
using LossFunctions: LossFunctions, Traits, deriv
using Statistics: mean

using Lux: Lux, LossFunctionImpl

const CRC = ChainRulesCore

function LossFunctionImpl.fused_agg(
    ::typeof(mean), lfn::Traits.Loss, x::AbstractArray, y::AbstractArray
)
    return LossFunctionImpl.fused_agg(sum, lfn, x, y) / length(x)
end

function LossFunctionImpl.fused_agg(::typeof(sum), lfn::Traits.Loss, x::Number, y::Number)
    return lfn(x, y)
end
function LossFunctionImpl.fused_agg(
    ::typeof(sum), lfn::Traits.Loss, x::AbstractArray, y::AbstractArray
)
    fast_scalar_indexing(x) && fast_scalar_indexing(y) && return sum(lfn, x, y)
    return sum(lfn.(x, y))
end

function CRC.rrule(
    ::CRC.RuleConfig{>:CRC.HasReverseMode},
    ::typeof(LossFunctionImpl.fused_agg),
    ::typeof(sum),
    lfn::Traits.Loss,
    x,
    y,
)
    ∇fused_agg = @closure Δ -> begin
        ∂x = @thunk deriv.(Ref(lfn), x, y) .* Δ
        return NoTangent(), NoTangent(), NoTangent(), ∂x, NoTangent()
    end
    return LossFunctionImpl.fused_agg(sum, lfn, x, y), ∇fused_agg
end

# macro hygiene issues
# EnzymeRules.@easy_rule(
#     LossFunctionImpl.fused_agg(fn::typeof(sum), lfn::Traits.Loss, x, y),
#     (@Constant, @Constant, deriv(lfn, x, y) .* Ω, @Constant)
# )

# COV_EXCL_START

function EnzymeRules.augmented_primal(
    cfg::EnzymeRules.RevConfigWidth{1},
    func::EnzymeCore.Const{typeof(LossFunctionImpl.fused_agg)},
    ::Type{<:EnzymeCore.Active},
    agg_f::EnzymeCore.Const{typeof(sum)},
    lfn::EnzymeCore.Const{<:Traits.Loss},
    x::EnzymeCore.Annotation{<:AbstractArray},
    y::EnzymeCore.Const,
)
    primal =
        EnzymeRules.needs_primal(cfg) ? func.val(agg_f.val, lfn.val, x.val, y.val) : nothing

    cache_x = EnzymeRules.overwritten(cfg)[4] ? copy(x.val) : nothing
    cache_y = EnzymeRules.overwritten(cfg)[5] ? copy(y.val) : nothing

    return EnzymeRules.AugmentedReturn(primal, nothing, (cache_x, cache_y))
end

function EnzymeRules.reverse(
    cfg::EnzymeRules.RevConfigWidth{1},
    ::EnzymeCore.Const{typeof(LossFunctionImpl.fused_agg)},
    dret::EnzymeCore.Active,
    (cache_x, cache_y),
    agg_f::EnzymeCore.Const{typeof(sum)},
    lfn::EnzymeCore.Const{<:Traits.Loss},
    x::EnzymeCore.Annotation{<:AbstractArray},
    y::EnzymeCore.Const,
)
    EnzymeRules.overwritten(cfg)[4] || (cache_x = x.val)
    EnzymeRules.overwritten(cfg)[5] || (cache_y = y.val)

    if !(typeof(x) <: EnzymeCore.Const)
        @. x.dval = deriv(lfn.val, cache_x, cache_y) * dret.val
    end

    return ntuple(Returns(nothing), 4)
end

# COV_EXCL_STOP

end

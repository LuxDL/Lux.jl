module LuxLossFunctionsExt

using ArrayInterface: fast_scalar_indexing
using ChainRulesCore: ChainRulesCore, NoTangent, @thunk
using EnzymeCore: EnzymeCore, EnzymeRules
using FastClosures: @closure
using LossFunctions: LossFunctions
using Statistics: mean

using Lux: Lux, LossFunctionImpl

const CRC = ChainRulesCore

function LossFunctionImpl.fused_agg(
    ::typeof(mean), lfn::LossFunctions.Traits.Loss, x::AbstractArray, y::AbstractArray
)
    return LossFunctionImpl.fused_agg(sum, lfn, x, y) / length(x)
end

function LossFunctionImpl.fused_agg(
    ::typeof(sum), lfn::LossFunctions.Traits.Loss, x::Number, y::Number
)
    return lfn(x, y)
end
function LossFunctionImpl.fused_agg(
    ::typeof(sum), lfn::LossFunctions.Traits.Loss, x::AbstractArray, y::AbstractArray
)
    fast_scalar_indexing(x) && fast_scalar_indexing(y) && return sum(lfn, x, y)
    return sum(lfn.(x, y))
end

function CRC.rrule(
    ::CRC.RuleConfig{>:CRC.HasReverseMode},
    ::typeof(LossFunctionImpl.fused_agg),
    ::typeof(sum),
    lfn::LossFunctions.Traits.Loss,
    x,
    y,
)
    ∇fused_agg = @closure Δ -> begin
        ∂x = @thunk LossFunctions.deriv.(Ref(lfn), x, y) .* Δ
        return NoTangent(), NoTangent(), NoTangent(), ∂x, NoTangent()
    end
    return LossFunctionImpl.fused_agg(sum, lfn, x, y), ∇fused_agg
end

# COV_EXCL_START

EnzymeRules.@easy_rule(
    LossFunctionImpl.fused_agg(fn::typeof(sum), lfn::LossFunctions.Traits.Loss, x, y),
    (@Constant, @Constant, LossFunctions.deriv(lfn, x, y) .* Ω, @Constant)
)

# COV_EXCL_STOP

end

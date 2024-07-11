# Non-differentiable
EnzymeRules.inactive(::typeof(__set_refval!), ::Any...) = nothing

# Loss Functions
function EnzymeRules.augmented_primal(
        cfg::EnzymeRules.ConfigWidth{1}, func::EnzymeCore.Const{typeof(__fused_agg)},
        ::Type{<:EnzymeCore.Active}, agg_f::EnzymeCore.Const{typeof(sum)},
        lfn::EnzymeCore.Const{<:LossFunctions.Traits.Loss},
        x::EnzymeCore.Annotation{<:AbstractArray}, y::EnzymeCore.Const)
    primal = EnzymeRules.needs_primal(cfg) ? func.val(agg_f.val, lfn.val, x.val, y.val) :
             nothing

    cache_x = EnzymeRules.overwritten(cfg)[4] ? copy(x.val) : nothing
    cache_y = EnzymeRules.overwritten(cfg)[5] ? copy(y.val) : nothing

    return EnzymeRules.AugmentedReturn(primal, nothing, (cache_x, cache_y))
end

function EnzymeRules.reverse(
        cfg::EnzymeRules.ConfigWidth{1}, ::EnzymeCore.Const{typeof(__fused_agg)},
        dret::EnzymeCore.Active, (cache_x, cache_y), agg_f::EnzymeCore.Const{typeof(sum)},
        lfn::EnzymeCore.Const{<:LossFunctions.Traits.Loss},
        x::EnzymeCore.Annotation{<:AbstractArray}, y::EnzymeCore.Const)
    EnzymeRules.overwritten(cfg)[4] || (cache_x = x.val)
    EnzymeRules.overwritten(cfg)[5] || (cache_y = y.val)

    if !(typeof(x) <: EnzymeCore.Const)
        @. x.dval = LossFunctions.deriv(lfn.val, cache_x, cache_y) * dret.val
    end

    return ntuple(Returns(nothing), 4)
end

module LuxZygoteExt

using ADTypes: AutoZygote
using ChainRulesCore: ChainRulesCore
using FastClosures: @closure
using Lux: Lux, DISABLE_AUTOMATIC_NESTED_AD_SWITCH
using Setfield: @set!
using Zygote: Zygote

const CRC = ChainRulesCore

function Lux.Experimental.compute_gradients(::AutoZygote, objective_function::F, data,
        ts::Lux.Experimental.TrainState) where {F}
    (loss, st, stats), back = Zygote.pullback(
        objective_function, ts.model, ts.parameters, ts.states, data)
    grads = back((one(loss), nothing, nothing))[2]
    @set! ts.states = st
    return grads, loss, stats, ts
end

# Nested AD Handling: Only for AbstractArray Inputs
# function CRC.rrule(::typeof(Zygote._pullback), ctx::Zygote.AContext,
#         model::Lux.StatefulLuxLayer, x::AbstractArray)
#     y, pb_f = Zygote._pullback(ctx, model, x)
#     ∇nested_pullback_default = Δ -> begin
#         @show Δ
#         error(2)
#     end
#     return (y, pb_f), ∇nested_pullback_default
# end

@inline __internal_gradient_capture(f::F, x, args...) where {F} = (first(Zygote.gradient(
    f, x, args...)),)

@inline function Zygote.gradient(
        f::Base.ComposedFunction{<:Lux.StatefulLuxLayer, F}, x::AbstractArray) where {F}
    return __internal_gradient_capture(@closure((x, ps)->f.outer(f.inner(x), ps)), x, ps)
end

@inline function Zygote.gradient(
        f::Base.ComposedFunction{F, <:Lux.StatefulLuxLayer}, x::AbstractArray) where {F}
    return __internal_gradient_capture(f, x, f.inner.ps)
end

@inline function Zygote.gradient(f::Lux.StatefulLuxLayer, x::AbstractArray)
    return __internal_gradient_capture(f, x, f.ps)
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__internal_gradient_capture), f::F, x::AbstractArray, ps) where {F}
    if !Lux._is_extension_loaded(Val(:ForwardDiff)) || DISABLE_AUTOMATIC_NESTED_AD_SWITCH
        if !DISABLE_AUTOMATIC_NESTED_AD_SWITCH
            @warn "Load ForwardDiff.jl for better nested AD handling." maxlog=1
        end
        # Use the AD itself for whatever reason
        y, pb_f = CRC.rrule_via_ad(cfg, Zygote.gradient, f, x, ps)
        return (first(y),), pb_f
    end

    y = __internal_gradient_capture(f, x, ps)
    ∇internal_gradient_capture = @closure Δ -> begin
        (Δ isa CRC.NoTangent || Δ isa CRC.ZeroTangent) &&
            return ntuple(Returns(CRC.NoTangent()), 4)
        Δ_ = reshape(CRC.unthunk(first(Δ)), size(x))
        ∂x, ∂ps = Lux.__forwarddiff_jvp(
            @closure((x, ps)->Zygote.gradient(f, x, ps)), x, Δ_, ps)
        return CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂ps
    end

    return y, ∇internal_gradient_capture
end

end

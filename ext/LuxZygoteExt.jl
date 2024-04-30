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

# Nested AD Handling
## Zygote.gradient call
@inline function Zygote.gradient(
        f::Base.ComposedFunction{<:Lux.StatefulLuxLayer, F}, x::AbstractArray) where {F}
    return Lux.__internal_ad_gradient_call(
        Zygote.gradient, @closure((x, ps)->f.outer(f.inner(x), ps)), x, f.inner.ps)
end

@inline function Zygote.gradient(
        f::Base.ComposedFunction{F, <:Lux.StatefulLuxLayer}, x::AbstractArray) where {F}
    return Lux.__internal_ad_gradient_call(Zygote.gradient, f, x, f.inner.ps)
end

@inline function Zygote.gradient(f::Lux.StatefulLuxLayer, x::AbstractArray)
    return Lux.__internal_ad_gradient_call(Zygote.gradient, f, x, f.ps)
end

## Zygote.jacobian call
@inline function __internal_jacobian_capture(f::F, x, args...) where {F}
    return Zygote.jacobian(@closure(x->f(x, args...)), x)
end

@inline function Zygote.jacobian(
        f::Base.ComposedFunction{<:Lux.StatefulLuxLayer, F}, x::AbstractArray) where {F}
    return __internal_jacobian_capture(@closure((x, ps)->f.outer(f.inner(x), ps)), x, ps)
end

@inline function Zygote.jacobian(
        f::Base.ComposedFunction{F, <:Lux.StatefulLuxLayer}, x::AbstractArray) where {F}
    return __internal_jacobian_capture(f, x, f.inner.ps)
end

@inline function Zygote.jacobian(f::Lux.StatefulLuxLayer, x::AbstractArray)
    return __internal_jacobian_capture(f, x, f.ps)
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__internal_jacobian_capture), f::F, x::AbstractArray, ps) where {F}
    if !Lux._is_extension_loaded(Val(:ForwardDiff)) || DISABLE_AUTOMATIC_NESTED_AD_SWITCH
        if !DISABLE_AUTOMATIC_NESTED_AD_SWITCH
            @warn "Load ForwardDiff.jl for better nested AD handling." maxlog=1
        end
        # Use the AD itself for whatever reason. This will fail most likely!
        y, pb_f = CRC.rrule_via_ad(cfg, Zygote.jacobian, f, x, ps)
        return y, pb_f
    end

    J = __internal_jacobian_capture(f, x, ps)
    ∇internal_jacobian_capture = Δ_ -> begin
        (Δ_ isa CRC.NoTangent || Δ_ isa CRC.ZeroTangent) &&
            return ntuple(Returns(CRC.NoTangent()), 4)

        Δ = Lux.__compactify_if_structured_matrix(only(J), CRC.unthunk(only(Δ_)))
        ∂x, ∂ps = mapreduce(Lux.__internal_add, enumerate(eachrow(Δ))) do (i, Δᵢ)
            __f = (x, p) -> sum(vec(f(x, p))[i:i])
            ∂xᵢ, ∂psᵢ = Lux.__forwarddiff_jvp(
                @closure((x, ps)->Zygote.gradient(__f, x, ps)), x, reshape(Δᵢ, size(x)), ps)
            return ∂xᵢ, ∂psᵢ
        end
        return CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂ps
    end

    return J, ∇internal_jacobian_capture
end

# Handle Weird Zygote shit
## Hope this doesn't get moved into extensions then we will have to create another file
@static if isdefined(Zygote, :ForwardDiff)
    using Zygote: ForwardDiff

    # Forward to a function that doesn't have this _pullback defined so that it triggers the
    # rrule
    function Zygote._pullback(cx::Zygote.AContext,
            ::typeof(ForwardDiff.jacobian),
            f::Union{Base.ComposedFunction{<:Any, <:Lux.StatefulLuxLayer},
                Base.ComposedFunction{<:Lux.StatefulLuxLayer, <:Any},
                Lux.StatefulLuxLayer},
            x::AbstractArray)
        return Zygote._pullback(
            cx, ForwardDiff.jacobian, f, x, ForwardDiff.JacobianConfig(f, x), Val(true))
    end

    function Zygote._pullback(cx::Zygote.AContext,
            ::typeof(ForwardDiff.gradient),
            f::Union{Base.ComposedFunction{<:Any, <:Lux.StatefulLuxLayer},
                Base.ComposedFunction{<:Lux.StatefulLuxLayer, <:Any},
                Lux.StatefulLuxLayer},
            x::AbstractArray)
        return Zygote._pullback(
            cx, ForwardDiff.gradient, f, x, ForwardDiff.GradientConfig(f, x), Val(true))
    end
end

end

module LuxForwardDiffExt

using ChainRulesCore: ChainRulesCore
using Lux: Lux, DISABLE_AUTOMATIC_NESTED_AD_SWITCH
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Functors: fmap

const CRC = ChainRulesCore

@inline Lux._is_extension_loaded(::Val{:ForwardDiff}) = true

@inline Lux.__partials(::Type{Tag}, x::AbstractArray, i) where {Tag} = ForwardDiff.partials.(
    Tag, x, i)
@inline Lux.__partials(::Type{Tag}, x::Tuple, i) where {Tag} = map(
    @closure(xᵢ->Lux.__partials(Tag, xᵢ, i)), x)
@inline Lux.__partials(::Type{Tag}, x::NamedTuple{F}, i) where {Tag, F} = NamedTuple{F}(map(
    @closure(xᵢ->Lux.__partials(Tag, xᵢ, i)), values(x)))
@inline Lux.__partials(::Type{Tag}, x::CRC.AbstractTangent, i) where {Tag} = Lux.__partials(
    Tag, CRC.backing(x), i)
@inline Lux.__partials(::Type{Tag}, ::Nothing, i) where {Tag} = nothing
@inline Lux.__partials(::Type{Tag}, x, i) where {Tag} = fmap(
    @closure(xᵢ->Lux.__partials(Tag, xᵢ, i)), x)

# This is not a general jvp code, but rather meant to be efficient for nested AD calls
function Lux.__forwarddiff_jvp(
        f::F, x::AbstractArray{xT}, Δx::AbstractArray{ΔxT}, ps) where {F, xT, ΔxT}
    T = promote_type(xT, ΔxT)
    Tag = typeof(ForwardDiff.Tag(f, T))
    partials = ForwardDiff.Partials{1, T}.(tuple.(Δx))
    x_dual = ForwardDiff.Dual{Tag, T, 1}.(x, reshape(partials, size(x)))
    y_dual, ps_dual = f(x_dual, ps)
    return Lux.__partials(Tag, y_dual, 1), Lux.__partials(Tag, ps_dual, 1)
end

# Capture ForwardDiff.jacobian call and replace it with forward over reverse mode AD
@inline function __updated_forwarddiff_config(
        ::ForwardDiff.JacobianConfig{T, V, N, D}, f::F,
        x::AbstractArray{V}) where {T, V, N, D, F}
    return ForwardDiff.JacobianConfig(f, x, ForwardDiff.Chunk{N}())
end

@inline function __updated_forwarddiff_config(
        ::ForwardDiff.GradientConfig{T, V, N, D}, f::F,
        x::AbstractArray{V}) where {T, V, N, D, F}
    return ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{N}())
end

# TODO: We can define multiple dispatches using meta programming to not construct these
#       intermediate configs, but that is kind of a micro-optimization, so we can handle
#       those later.
@inline function __internal_gradient_capture(
        f::F, cfg::ForwardDiff.GradientConfig, chk::Val, x, args...) where {F}
    # Here we can't really pass in the actual config because we modify the internal function
    __f = @closure(x->f(x, args...))
    return ForwardDiff.gradient(__f, x, __updated_forwarddiff_config(cfg, __f, x), chk)
end

@inline function ForwardDiff.gradient(
        f::Base.ComposedFunction{<:Lux.StatefulLuxLayer, F}, x::AbstractArray,
        cfg::ForwardDiff.GradientConfig=ForwardDiff.GradientConfig(f, x),
        check::Val=Val(true)) where {F}
    return __internal_gradient_capture(
        @closure((x, ps)->f.outer(f.inner(x), ps)), cfg, check, x, ps)
end

@inline function ForwardDiff.gradient(
        f::Base.ComposedFunction{F, <:Lux.StatefulLuxLayer}, x::AbstractArray,
        cfg::ForwardDiff.GradientConfig=ForwardDiff.GradientConfig(f, x),
        check::Val=Val(true)) where {F}
    return __internal_gradient_capture(f, cfg, check, x, f.inner.ps)
end

@inline function ForwardDiff.gradient(f::Lux.StatefulLuxLayer, x::AbstractArray,
        cfg::ForwardDiff.GradientConfig=ForwardDiff.GradientConfig(f, x),
        check::Val=Val(true))
    return __internal_gradient_capture(f, cfg, check, x, f.ps)
end

function CRC.rrule(
        cfg::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(__internal_gradient_capture),
        f::F, jc_cfg::ForwardDiff.GradientConfig, chk::Val, x::AbstractArray, ps) where {F}
    if DISABLE_AUTOMATIC_NESTED_AD_SWITCH
        y, pb_f = CRC.rrule_via_ad(cfg, ForwardDiff.gradient, Base.Fix2(f, ps), x)
        ∇internal_jacobian_capture_noswitch = Δ -> begin
            @warn "Nested AD switch is disabled for `ForwardDiff.gradient`. If used with \
                   an outer `Zygote.gradient` call, the gradients wrt parameters `ps` will \
                   be dropped. Enable nested AD switching to get the correct (full) \
                   gradients." maxlog=1
            _, _, ∂x = pb_f(Δ)
            return (CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(),
                CRC.NoTangent(), ∂x, CRC.NoTangent())
        end
        return y, ∇internal_jacobian_capture_noswitch
    end

    g = __internal_gradient_capture(f, jc_cfg, chk, x, ps)
    ∇internal_gradient_capture = Δ_ -> begin
        (Δ_ isa CRC.NoTangent || Δ_ isa CRC.ZeroTangent) &&
            return ntuple(Returns(CRC.NoTangent()), 6)

        Δ = reshape(CRC.unthunk(Δ_), size(x))
        ∂x, ∂ps = Lux.__forwarddiff_jvp(x, Δ, ps) do x, ps
            y, pb_f = CRC.rrule_via_ad(cfg, f, x, ps)
            return pb_f(one(y))[2:3]
        end
        return (CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂ps)
    end
    return g, ∇internal_gradient_capture
end

@inline function __internal_jacobian_capture(
        f::F, cfg::ForwardDiff.JacobianConfig, chk::Val, x, args...) where {F}
    # Here we can't really pass in the actual config because we modify the internal function
    __f = @closure(x->f(x, args...))
    return ForwardDiff.jacobian(__f, x, __updated_forwarddiff_config(cfg, __f, x), chk)
end

@inline function ForwardDiff.jacobian(
        f::Base.ComposedFunction{<:Lux.StatefulLuxLayer, F}, x::AbstractArray,
        cfg::ForwardDiff.JacobianConfig=ForwardDiff.JacobianConfig(f, x),
        check::Val=Val(true)) where {F}
    return __internal_jacobian_capture(
        @closure((x, ps)->f.outer(f.inner(x), ps)), cfg, check, x, ps)
end

@inline function ForwardDiff.jacobian(
        f::Base.ComposedFunction{F, <:Lux.StatefulLuxLayer}, x::AbstractArray,
        cfg::ForwardDiff.JacobianConfig=ForwardDiff.JacobianConfig(f, x),
        check::Val=Val(true)) where {F}
    return __internal_jacobian_capture(f, cfg, check, x, f.inner.ps)
end

@inline function ForwardDiff.jacobian(f::Lux.StatefulLuxLayer, x::AbstractArray,
        cfg::ForwardDiff.JacobianConfig=ForwardDiff.JacobianConfig(f, x),
        check::Val=Val(true))
    return __internal_jacobian_capture(f, cfg, check, x, f.ps)
end

function CRC.rrule(
        cfg::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(__internal_jacobian_capture),
        f::F, jc_cfg::ForwardDiff.JacobianConfig, chk::Val, x::AbstractArray, ps) where {F}
    if DISABLE_AUTOMATIC_NESTED_AD_SWITCH
        y, pb_f = CRC.rrule_via_ad(cfg, ForwardDiff.jacobian, Base.Fix2(f, ps), x)
        ∇internal_jacobian_capture_noswitch = Δ -> begin
            @warn "Nested AD switch is disabled for `ForwardDiff.jacobian`. If used with \
                   an outer `Zygote.gradient` call, the gradients wrt parameters `ps` will \
                   be dropped. Enable nested AD switching to get the correct (full) \
                   gradients." maxlog=1
            _, _, ∂x = pb_f(Δ)
            return (CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(),
                CRC.NoTangent(), ∂x, CRC.NoTangent())
        end
        return y, ∇internal_jacobian_capture_noswitch
    end

    J = __internal_jacobian_capture(f, jc_cfg, chk, x, ps)

    ∇internal_jacobian_capture = Δ_ -> begin
        (Δ_ isa CRC.NoTangent || Δ_ isa CRC.ZeroTangent) &&
            return ntuple(Returns(CRC.NoTangent()), 6)

        Δ = reshape(CRC.unthunk(Δ_), size(J))
        ∂x, ∂ps = mapreduce(Lux.__internal_add, enumerate(eachrow(Δ))) do (i, Δᵢ)
            __f = (x, p) -> sum(vec(f(x, p))[i:i])
            __gradient_fn = (x, ps) -> begin
                y, pb_f = CRC.rrule_via_ad(cfg, __f, x, ps)
                return pb_f(one(y))[2:3]
            end
            return Lux.__forwarddiff_jvp(__gradient_fn, x, reshape(Δᵢ, size(x)), ps)
        end
        return (CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂ps)
    end

    return J, ∇internal_jacobian_capture
end

end

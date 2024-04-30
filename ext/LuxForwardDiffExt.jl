module LuxForwardDiffExt

using ChainRulesCore: ChainRulesCore
using Lux: Lux
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Functors: fmap

const CRC = ChainRulesCore

@inline Lux._is_extension_loaded(::Val{:ForwardDiff}) = true

@inline function Lux.__partials(::Type{Tag}, x, i) where {Tag}
    x isa AbstractArray && return ForwardDiff.partials.(Tag, x, i)
    map_fn = @closure(xᵢ->Lux.__partials(Tag, xᵢ, i))
    x isa Tuple && return map(map_fn, x)
    x isa NamedTuple && return NamedTuple{keys(x)}(map(map_fn, values(x)))
    x isa CRC.AbstractTangent && return Lux.__partials(Tag, CRC.backing(x), i)
    x === nothing && return nothing
    return fmap(map_fn, x)
end

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
for cfg in (:JacobianConfig, :GradientConfig)
    @eval @inline function __updated_forwarddiff_config(
            ::ForwardDiff.$(cfg){T, V, N, D}, f::F,
            x::AbstractArray{V}) where {T, V, N, D, F}
        return ForwardDiff.$(cfg)(f, x, ForwardDiff.Chunk{N}())
    end
end

for fType in Lux.AD_CONVERTIBLE_FUNCTIONS, type in (:Gradient, :Jacobian)
    cfgname = Symbol(type, :Config)
    fname = Symbol(lowercase(string(type)))
    internal_fname = Symbol(:__internal_forwarddiff_, fname)

    @eval begin
        @inline function ForwardDiff.$(fname)(f::$fType, x::AbstractArray,
                cfg::ForwardDiff.$(cfgname)=ForwardDiff.$(cfgname)(f, x),
                chk::Val=Val(true))
            f_internal, ps = Lux.__rewrite_ad_call(f)
            return $(internal_fname)(f_internal, cfg, chk, x, ps)
        end
    end
end

for type in (:Gradient, :Jacobian)
    cfgname = Symbol(type, :Config)
    fname = Symbol(lowercase(string(type)))
    internal_fname = Symbol(:__internal_forwarddiff_, fname)

    @eval @inline function $(internal_fname)(
            f::F, cfg::ForwardDiff.$(cfgname), chk::Val, x::AbstractArray, y) where {F}
        __f = Base.Fix2(f, y)
        return ForwardDiff.$(fname)(__f, x, __updated_forwarddiff_config(cfg, __f, x), chk)
    end

    rrule_call = if type == :Gradient
        :((res, pb_f) = CRC.rrule_via_ad(
            cfg, Lux.__internal_ad_gradient_call, grad_fn, f, x, y))
    else
        :((res, pb_f) = CRC.rrule_via_ad(
            cfg, Lux.__internal_ad_jacobian_call, ForwardDiff.$(fname), grad_fn, f, x, y))
    end
    ret_expr = type == :Gradient ? :(only(res)) : :(res)
    @eval begin
        function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
                ::typeof($(internal_fname)), f::F, jc_cfg::ForwardDiff.$(cfgname),
                chk::Val, x::AbstractArray, y) where {F}
            grad_fn = (f_internal, x, args...) -> begin
                res, ∂f = CRC.rrule_via_ad(cfg, f_internal, x, args...)
                return ∂f(one(res))[2:end]
            end

            $(rrule_call)
            ∇internal_nested_ad_capture = Δ -> begin
                ∂x, ∂y = pb_f(tuple(Δ))[(end - 1):end]
                return (ntuple(Returns(CRC.NoTangent()), 4)..., ∂x, ∂y)
            end
            return $(ret_expr), ∇internal_nested_ad_capture
        end
    end
end

end

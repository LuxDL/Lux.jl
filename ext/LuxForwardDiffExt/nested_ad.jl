for fType in Lux.AD_CONVERTIBLE_FUNCTIONS
    @eval @inline function Lux.__jacobian_vector_product_impl(
            f::$(fType), ::AutoForwardDiff, x, u)
        f_internal, y = Lux.__rewrite_ad_call(f)
        return __jacobian_vector_product_ad_impl(f_internal, x, u, y)
    end
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__jacobian_vector_product_ad_impl), f::F, x, u, y) where {F}
    res = __jacobian_vector_product_ad_impl(f, x, u, y)

    pullback_fn = (f_internal, x, args...) -> begin
        res, ∂f = CRC.rrule_via_ad(cfg, f_internal, x, args...)
        ∂f_internal(Δ) = ∂f(Δ)[2:end]
        return res, ∂f_internal
    end

    ∇internal_nested_pushforward_capture = Δ -> begin
        _, pb_f = CRC.rrule_via_ad(
            cfg, Lux.__internal_ad_pullback_call, pullback_fn, f, x, y, Δ)
        _, _, _, ∂x, ∂y, _ = pb_f(u)
        return CRC.NoTangent(), CRC.NoTangent(), ∂x, CRC.NoTangent(), ∂y
    end

    return res, ∇internal_nested_pushforward_capture
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
            f_internal, y = Lux.__rewrite_ad_call(f)
            return $(internal_fname)(f_internal, cfg, chk, x, y)
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

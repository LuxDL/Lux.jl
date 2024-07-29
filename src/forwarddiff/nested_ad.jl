# Capture ForwardDiff.jacobian/gradient call and replace it with forward over reverse mode
# AD
for cfg in (:JacobianConfig, :GradientConfig)
    @eval function __updated_forwarddiff_config(::ForwardDiff.$(cfg){T, V, N, D}, f::F,
            x::AbstractArray{V}) where {T, V, N, D, F}
        return ForwardDiff.$(cfg)(f, x, ForwardDiff.Chunk{N}())
    end
end

for fType in AD_CONVERTIBLE_FUNCTIONS, type in (:Gradient, :Jacobian)
    cfgname = Symbol(type, :Config)
    fname = Symbol(lowercase(string(type)))
    internal_fname = Symbol(:__internal_forwarddiff_, fname)

    @eval function ForwardDiff.$(fname)(f::$fType, x::AbstractArray,
            cfg::ForwardDiff.$(cfgname)=ForwardDiff.$(cfgname)(f, x), chk::Val=Val(true))
        f_internal, y = __rewrite_ad_call(f)
        return $(internal_fname)(f_internal, cfg, chk, x, y)
    end
end

for type in (:Gradient, :Jacobian)
    cfgname = Symbol(type, :Config)
    fname = Symbol(lowercase(string(type)))
    internal_fname = Symbol(:__internal_forwarddiff_, fname)

    @eval function $(internal_fname)(
            f::F, cfg::ForwardDiff.$(cfgname), chk::Val, x::AbstractArray, y) where {F}
        __f = Base.Fix2(f, y)
        return ForwardDiff.$(fname)(__f, x, __updated_forwarddiff_config(cfg, __f, x), chk)
    end

    rrule_call = if type == :Gradient
        :((res, pb_f) = CRC.rrule_via_ad(cfg, __internal_ad_gradient_call, grad_fn, f, x, y))
    else
        :((res, pb_f) = CRC.rrule_via_ad(
            cfg, __internal_ad_jacobian_call, ForwardDiff.$(fname), grad_fn, f, x, y))
    end
    ret_expr = type == :Gradient ? :(only(res)) : :(res)
    @eval begin
        function CRC.rrule(
                func::typeof($(internal_fname)), f::F, jc_cfg::ForwardDiff.$(cfgname),
                chk::Val, x::AbstractArray, y) where {F}
            return CRC.rrule_via_ad(_zygote_rule_config(), func, f, jc_cfg, chk, x, y)
        end

        function CRC.rrule(
                cfg::RuleConfig{>:HasReverseMode}, ::typeof($(internal_fname)), f::F,
                jc_cfg::ForwardDiff.$(cfgname), chk::Val, x::AbstractArray, y) where {F}
            grad_fn = let cfg = cfg
                (f_internal, x, args...) -> begin
                    res, ∂f = CRC.rrule_via_ad(cfg, f_internal, x, args...)
                    return ∂f(one(res))[2:end]
                end
            end

            $(rrule_call)
            ∇internal_nested_ad_capture = let pb_f = pb_f
                Δ -> begin
                    ∂x, ∂y = pb_f(tuple(Δ))[(end - 1):end]
                    return (ntuple(Returns(NoTangent()), 4)..., ∂x, ∂y)
                end
            end
            return $(ret_expr), ∇internal_nested_ad_capture
        end
    end
end

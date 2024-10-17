for cfg in (:JacobianConfig, :GradientConfig)
    @eval function updated_forwarddiff_config(::ForwardDiff.$(cfg){T, V, N, D}, f::F,
            x::AbstractArray{V}) where {T, V, N, D, F}
        return ForwardDiff.$(cfg)(f, x, ForwardDiff.Chunk{N}())
    end
end

for fType in AD_CONVERTIBLE_FUNCTIONS, type in (:Gradient, :Jacobian)

    cfgname = Symbol(type, :Config)
    fname = Symbol(lowercase(string(type)))
    internal_fname = Symbol(:forwarddiff_, fname)

    @eval function ForwardDiff.$(fname)(f::$fType, x::AbstractArray,
            cfg::ForwardDiff.$(cfgname)=ForwardDiff.$(cfgname)(f, x), chk::Val=Val(true))
        f̂, y = rewrite_autodiff_call(f)
        return $(internal_fname)(f̂, cfg, chk, x, y)
    end
end

for type in (:Gradient, :Jacobian)
    cfgname = Symbol(type, :Config)
    fname = Symbol(lowercase(string(type)))
    internal_fname = Symbol(:forwarddiff_, fname)

    @eval function $(internal_fname)(
            f::F, cfg::ForwardDiff.$(cfgname), chk::Val, x::AbstractArray, y) where {F}
        f̂ = Base.Fix2(f, y)
        return ForwardDiff.$(fname)(f̂, x, updated_forwarddiff_config(cfg, f̂, x), chk)
    end

    rrule_call = if type == :Gradient
        :((res, pb_f) = CRC.rrule_via_ad(cfg, autodiff_gradient, grad_fn, f, x, y))
    else
        :((res,
            pb_f) = CRC.rrule_via_ad(
            cfg, autodiff_jacobian, ForwardDiff.$(fname), grad_fn, f, x, y))
    end
    ret_expr = type == :Gradient ? :(only(res)) : :(res)
    @eval begin
        function CRC.rrule(
                cfg::RuleConfig{>:HasReverseMode}, ::typeof($(internal_fname)), f::F,
                jc_cfg::ForwardDiff.$(cfgname), chk::Val, x::AbstractArray, y) where {F}
            grad_fn = let cfg = cfg
                (f̂, x, args...) -> begin
                    res, ∂f = CRC.rrule_via_ad(cfg, f̂, x, args...)
                    return ∂f(one(res))[2:end]
                end
            end

            $(rrule_call)
            ∇forwarddiff_ad = let pb_f = pb_f
                Δ -> begin
                    ∂x, ∂y = pb_f(tuple(Δ))[(end - 1):end]
                    𝒫x, 𝒫y = CRC.ProjectTo(x), CRC.ProjectTo(y)
                    return (ntuple(Returns(NoTangent()), 4)..., 𝒫x(∂x), 𝒫y(∂y))
                end
            end
            return $(ret_expr), ∇forwarddiff_ad
        end
    end
end

# This is not a general jvp code, but rather meant to be efficient for nested AD calls
function Lux.__forwarddiff_jvp(f::F, x, Δx, y) where {F}
    T = promote_type(Lux.recursive_eltype(x), Lux.recursive_eltype(Δx))
    Tag = typeof(ForwardDiff.Tag(f, T))
    res1_dual, res2_dual = f(Lux.__dualify(Tag, T, x, Δx), y)
    return (Lux.__partials(Tag, res1_dual, 1), Lux.__partials(Tag, res2_dual, 1))
end

# jvp
function Lux.__jacobian_vector_product_impl(f::F, ::AutoForwardDiff, x, u) where {F}
    T = promote_type(Lux.recursive_eltype(x), Lux.recursive_eltype(u))
    Tag = typeof(ForwardDiff.Tag(f, T))
    y_dual = f(Lux.__dualify(Tag, T, x, u))
    return Lux.__partials(Tag, y_dual, 1)
end

function __jacobian_vector_product_ad_impl(f::F, x, u, y) where {F}
    return Lux.__jacobian_vector_product_impl(Base.Fix2(f, y), AutoForwardDiff(), x, u)
end

## Nested AD for JVP
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

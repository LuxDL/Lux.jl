# This is not a general jvp code, but rather meant to be efficient for nested AD calls
function __forwarddiff_jvp(f::F, x, Δx, y) where {F}
    T = promote_type(recursive_eltype(x), recursive_eltype(Δx))
    Tag = typeof(ForwardDiff.Tag(f, T))
    res1_dual, res2_dual = f(__dualify(Tag, T, x, Δx), y)
    return (__partials(Tag, res1_dual, 1), __partials(Tag, res2_dual, 1))
end

# jvp
function __jacobian_vector_product_impl(f::F, ::AutoForwardDiff, x, u) where {F}
    T = promote_type(recursive_eltype(x), recursive_eltype(u))
    Tag = typeof(ForwardDiff.Tag(f, T))
    y_dual = f(__dualify(Tag, T, x, u))
    return __partials(Tag, y_dual, 1)
end

function __jacobian_vector_product_ad_impl(f::F, x, u, y) where {F}
    return __jacobian_vector_product_impl(Base.Fix2(f, y), AutoForwardDiff(), x, u)
end

## Nested AD for JVP
for fType in AD_CONVERTIBLE_FUNCTIONS
    @eval function __jacobian_vector_product_impl(f::$(fType), ::AutoForwardDiff, x, u)
        f_internal, y = __rewrite_ad_call(f)
        return __jacobian_vector_product_ad_impl(f_internal, x, u, y)
    end
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode},
        ::typeof(__jacobian_vector_product_ad_impl), f::F, x, u, y) where {F}
    res = __jacobian_vector_product_ad_impl(f, x, u, y)

    pullback_fn = let cfg = cfg
        (f_internal, x, args...) -> begin
            internal_res, ∂f = CRC.rrule_via_ad(cfg, f_internal, x, args...)
            ∂f_internal = let ∂f = ∂f
                Δ -> ∂f(Δ)[2:end]
            end
            return internal_res, ∂f_internal
        end
    end

    ∇nested_jvp = let pullback_fn = pullback_fn, f = f, x = x, y = y, u = u, cfg = cfg
        Δ -> begin
            _, pb_f = CRC.rrule_via_ad(
                cfg, __internal_ad_pullback_call, pullback_fn, f, x, y, Δ)
            _, _, _, ∂x, ∂y, _ = pb_f(u)
            return NoTangent(), NoTangent(), ∂x, NoTangent(), ∂y
        end
    end

    return res, ∇nested_jvp
end

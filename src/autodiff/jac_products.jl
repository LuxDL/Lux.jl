# VJP Implementation
function vector_jacobian_product(f::F, backend::AbstractADType, x, u) where {F}
    return vector_jacobian_product_impl(f, backend, x, u)
end

for fType in AD_CONVERTIBLE_FUNCTIONS
    @eval function vector_jacobian_product(f::$(fType), backend::AbstractADType, x, u)
        f̂, y = rewrite_autodiff_call(f)
        return vector_jacobian_product_impl(f̂, backend, x, u, y)
    end
end

function vector_jacobian_product_impl(f::F, backend::AbstractADType, x, u, y) where {F}
    return vector_jacobian_product_impl(Base.Fix2(f, y), backend, x, u)
end

# JVP Implementation
function jacobian_vector_product(f::F, backend::AbstractADType, x, u) where {F}
    return jacobian_vector_product_impl(f, backend, x, u)
end

for fType in AD_CONVERTIBLE_FUNCTIONS
    @eval function jacobian_vector_product(f::$(fType), backend::AbstractADType, x, u)
        f̂, y = rewrite_autodiff_call(f)
        return jacobian_vector_product_impl(f̂, backend, x, u, y)
    end
end

function jacobian_vector_product_impl(f::F, backend::AbstractADType, x, u, y) where {F}
    return jacobian_vector_product_impl(Base.Fix2(f, y), backend, x, u)
end

function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(jacobian_vector_product_impl),
        f::F, backend::AbstractADType, x, u, y) where {F}
    res = jacobian_vector_product_impl(f, backend, x, u, y)

    pullback_fn = let cfg = cfg
        (f̂, x, args...) -> begin
            internal_res, ∂f = CRC.rrule_via_ad(cfg, f̂, x, args...)
            ∂f_internal = let ∂f = ∂f
                Δ -> ∂f(Δ)[2:end]
            end
            return internal_res, ∂f_internal
        end
    end

    ∇jacvec_product = let pullback_fn = pullback_fn, f = f, x = x, y = y, u = u, cfg = cfg
        Δ -> begin
            _, ∇autodiff_pullback = CRC.rrule_via_ad(
                cfg, autodiff_pullback, pullback_fn, f, x, y, Δ)
            _, _, _, ∂x, ∂y, _ = ∇autodiff_pullback(u)
            return NoTangent(), NoTangent(), NoTangent(), ∂x, NoTangent(), ∂y
        end
    end

    return res, ∇jacvec_product
end

# ForwardDiff.jl Implementation
function jacobian_vector_product_impl(f::F, ::AutoForwardDiff, x, u) where {F}
    T = promote_type(Lux.recursive_eltype(x), Lux.recursive_eltype(u))
    Tag = typeof(ForwardDiff.Tag(f, T))
    y_dual = f(construct_duals(Tag, T, x, u))
    return extract_partials(Tag, y_dual, 1)
end

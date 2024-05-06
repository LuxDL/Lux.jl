# This is not a general jvp code, but rather meant to be efficient for nested AD calls
function Lux.__forwarddiff_jvp(f::F, x, Δx, y) where {F}
    T = promote_type(Lux.__recursive_eltype(x), Lux.__recursive_eltype(Δx))
    Tag = typeof(ForwardDiff.Tag(f, T))
    res1_dual, res2_dual = f(Lux.__dualify(Tag, T, x, Δx), y)
    return (Lux.__partials(Tag, res1_dual, 1), Lux.__partials(Tag, res2_dual, 1))
end

# jvp
function Lux.__jacobian_vector_product_impl(f::F, ::AutoForwardDiff, x, u) where {F}
    T = promote_type(Lux.__recursive_eltype(x), Lux.__recursive_eltype(u))
    Tag = typeof(ForwardDiff.Tag(f, T))
    y_dual = f(Lux.__dualify(Tag, T, x, u))
    return Lux.__partials(Tag, y_dual, 1)
end

function __jacobian_vector_product_ad_impl(f::F, x, u, y) where {F}
    return Lux.__jacobian_vector_product_impl(Base.Fix2(f, y), AutoForwardDiff(), x, u)
end

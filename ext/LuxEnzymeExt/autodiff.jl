# VJPs

function _vector_jacobian_product_impl(f::F, ad::AutoEnzyme, x, v, extra_args...) where {F}
    ad = Utils.normalize_autoenzyme_mode(Reverse, ad)
    @assert ADTypes.mode(ad) isa ADTypes.ReverseMode "VJPs are only supported in reverse \
                                                      mode"
    dx = fmap(zero, x; exclude=isleaf)
    Enzyme.autodiff(
        ad.mode,
        Utils.annotate_enzyme_function(ad, OOPFunctionWrapper(f)),
        Duplicated(fmap(similar, v; exclude=isleaf), fmap(copy, v; exclude=isleaf)),
        Duplicated(x, dx),
        extra_args...,
    )
    return dx
end

function AutoDiffInternalImpl.vector_jacobian_product_impl(
    f::F, ad::AutoEnzyme, x, v, p
) where {F}
    return _vector_jacobian_product_impl(f, ad, x, v, Const(p))
end

function AutoDiffInternalImpl.vector_jacobian_product_impl(
    f::F, ad::AutoEnzyme, x, v
) where {F}
    return _vector_jacobian_product_impl(f, ad, x, v)
end

# JVPs

function _jacobian_vector_product_impl(f::F, ad::AutoEnzyme, x, u, extra_args...) where {F}
    ad = Utils.normalize_autoenzyme_mode(Forward, ad)
    @assert ADTypes.mode(ad) isa ADTypes.ForwardMode "JVPs are only supported in forward \
                                                      mode"
    return only(
        Enzyme.autodiff(
            ad.mode, Utils.annotate_enzyme_function(ad, f), Duplicated(x, u), extra_args...
        ),
    )
end

function AutoDiffInternalImpl.jacobian_vector_product_impl(
    f::F, ad::AutoEnzyme, x, u, p
) where {F}
    return _jacobian_vector_product_impl(f, ad, x, u, Const(p))
end

function AutoDiffInternalImpl.jacobian_vector_product_impl(
    f::F, ad::AutoEnzyme, x, u
) where {F}
    return _jacobian_vector_product_impl(f, ad, x, u)
end

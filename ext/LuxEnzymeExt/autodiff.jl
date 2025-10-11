# VJPs

function AutoDiffInternalImpl.vector_jacobian_product_impl(
    f::F, ad::AutoEnzyme, x, v, p
) where {F}
    ad = normalize_backend(False(), ad)
    @assert ADTypes.mode(ad) isa ADTypes.ReverseMode "VJPs are only supported in reverse \
                                                      mode"
    dx = fmap(copy, x; exclude=isleaf)
    Enzyme.autodiff(
        ad.mode,
        annotate_function(ad, OOPFunctionWrapper(f)),
        Duplicated(fmap(similar, v; exclude=isleaf), fmap(copy, v; exclude=isleaf)),
        Duplicated(x, dx),
        Const(p),
    )
    return dx
end

function AutoDiffInternalImpl.vector_jacobian_product_impl(
    f::F, ad::AutoEnzyme, x, v
) where {F}
    ad = normalize_backend(False(), ad)
    @assert ADTypes.mode(ad) isa ADTypes.ReverseMode "VJPs are only supported in reverse \
                                                      mode"
    dx = fmap(copy, x; exclude=isleaf)
    Enzyme.autodiff(
        ad.mode,
        annotate_function(ad, OOPFunctionWrapper(f)),
        Duplicated(fmap(similar, v; exclude=isleaf), fmap(copy, v; exclude=isleaf)),
        Duplicated(x, dx),
    )
    return dx
end

# JVPs

function AutoDiffInternalImpl.jacobian_vector_product_impl(
    f::F, ad::AutoEnzyme, x, u, p
) where {F}
    ad = normalize_backend(True(), ad)
    @assert ADTypes.mode(ad) isa ADTypes.ForwardMode "JVPs are only supported in forward \
                                                      mode"
    return only(
        Enzyme.autodiff(ad.mode, annotate_function(ad, f), Duplicated(x, u), Const(p))
    )
end

function AutoDiffInternalImpl.jacobian_vector_product_impl(
    f::F, ad::AutoEnzyme, x, u
) where {F}
    ad = normalize_backend(True(), ad)
    @assert ADTypes.mode(ad) isa ADTypes.ForwardMode "JVPs are only supported in forward \
                                                      mode"
    return only(Enzyme.autodiff(ad.mode, annotate_function(ad, f), Duplicated(x, u)))
end

function Lux.AutoDiffInternalImpl.jacobian_vector_product_impl(
        f::F, ad::AutoEnzyme, x, u, p) where {F}
    ad = normalize_backend(True(), ad)
    @assert ADTypes.mode(ad) isa ForwardMode "JVPs are only supported in forward mode."
    return only(
        Enzyme.autodiff(ad.mode, annotate_function(ad, f), Duplicated(x, u), Const(p))
    )
end

function Lux.AutoDiffInternalImpl.jacobian_vector_product_impl(
        f::F, ad::AutoEnzyme, x, u) where {F}
    ad = normalize_backend(True(), ad)
    @assert ADTypes.mode(ad) isa ForwardMode "JVPs are only supported in forward mode."
    return only(Enzyme.autodiff(ad.mode, annotate_function(ad, f), Duplicated(x, u)))
end

function Lux.AutoDiffInternalImpl.vector_jacobian_product_impl(
        f::F, ad::AutoEnzyme, x, v, p) where {F}
    ad = normalize_backend(False(), ad)
    @assert ADTypes.mode(ad) isa ReverseMode "VJPs are only supported in reverse mode."
    dx = zero(x)
    # XXX: without the copy it overwrites the `v` with zeros
    Enzyme.autodiff(ad.mode, annotate_function(ad, OOPFunctionWrapper(f)),
        Duplicated(similar(v), copy(v)), Duplicated(x, dx), Const(p))
    return dx
end

function Lux.AutoDiffInternalImpl.vector_jacobian_product_impl(
        f::F, ad::AutoEnzyme, x, v) where {F}
    ad = normalize_backend(False(), ad)
    @assert ADTypes.mode(ad) isa ReverseMode "VJPs are only supported in reverse mode."
    dx = zero(x)
    # XXX: without the copy it overwrites the `v` with zeros
    Enzyme.autodiff(ad.mode, annotate_function(ad, OOPFunctionWrapper(f)),
        Duplicated(similar(v), copy(v)), Duplicated(x, dx))
    return dx
end

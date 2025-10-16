function Lux.AutoDiffInternalImpl.batched_jacobian_impl(
    f::F, ad::Lux.Training.ReactantBackend, x
) where {F}
    ad = Utils.normalize_autoenzyme_mode(EnzymeCore.Forward, ad.ad)
    if ADTypes.mode(ad) isa ADTypes.ReverseMode
        return _batched_jacobian_reverse_impl(f, ad, x)
    else
        return _batched_jacobian_forward_impl(f, ad, x)
    end
end

struct ApplyWithReshape{F,SZ}
    f::F
    sz::SZ
end

(f::ApplyWithReshape)(x) = reshape(f.f(reshape(x, f.sz)), :, size(x, ndims(x)))

function (f::ApplyWithReshape)(y, x)
    res = f.f(reshape(x, f.sz))
    copyto!(y, reshape(res, size(y)))
    return nothing
end

function _batched_jacobian_reverse_impl(f::F, ad::AutoEnzyme, x::AbstractArray) where {F}
    y = f(x)
    @assert y isa AbstractArray
    if ndims(y) ≤ 1 || size(y, ndims(y)) != size(x, ndims(x))
        throw(AssertionError("`batched_jacobian` only supports batched outputs \
                              (ndims(y) > 1) && size(y, ndims(y)) == size(x, ndims(x))."))
    end

    f′ = ApplyWithReshape(f, size(x))

    y = Utils.contiguous(reshape(y, :, size(y, ndims(y))))
    dy = repeat(
        reshape(
            Reactant.promote_to(
                TracedRArray{Reactant.unwrapped_eltype(y),2}, LinearAlgebra.I(size(y, 1))
            ),
            size(y, 1),
            1,
            size(y, 1),
        ),
        1,
        size(y, 2),
        1,
    )
    dy = Utils.contiguous(dy)

    x = Utils.contiguous(reshape(x, :, size(x, ndims(x))))
    dx = similar(x, size(x, 1), size(x, 2), size(y, 1))
    fill!(dx, false)

    Enzyme.autodiff(
        ad.mode,
        Utils.annotate_enzyme_function(ad, f′),
        Reactant.StackedBatchDuplicated(y, dy),
        Reactant.StackedBatchDuplicated(x, dx),
    )

    return permutedims(dx, (3, 1, 2))
end

function _batched_jacobian_forward_impl(f::F, ad::AutoEnzyme, x::AbstractArray) where {F}
    f′ = ApplyWithReshape(f, size(x))
    x = Utils.contiguous(reshape(x, :, size(x, ndims(x))))

    bx = repeat(
        reshape(
            Reactant.promote_to(
                TracedRArray{Reactant.unwrapped_eltype(x),2}, LinearAlgebra.I(size(x, 1))
            ),
            size(x, 1),
            1,
            size(x, 1),
        ),
        1,
        size(x, 2),
        1,
    )
    bx = Utils.contiguous(bx)

    return stack(
        only(
            Enzyme.autodiff(
                ad.mode,
                Utils.annotate_enzyme_function(ad, f′),
                Reactant.StackedBatchDuplicated(x, bx),
            ),
        );
        dims=2,
    )
end

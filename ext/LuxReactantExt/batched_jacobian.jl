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

function _check_validity_for_batched_jacobian(f::F, x::AbstractArray) where {F}
    y = f(x)
    @assert y isa AbstractArray
    B = size(y, ndims(y))
    if ndims(y) ≤ 1 || B != size(x, ndims(x))
        throw(AssertionError("`batched_jacobian` only supports batched outputs \
                              (ndims(y) > 1) && size(y, ndims(y)) == size(x, ndims(x))."))
    end
    return y, B
end

function _batched_jacobian_reverse_impl(f::F, ad::AutoEnzyme, x::AbstractArray) where {F}
    y, B = _check_validity_for_batched_jacobian(f, x)
    f′ = ApplyWithReshape(f, size(x))

    y = Utils.contiguous(reshape(y, :, B))
    dy = Utils.contiguous(
        repeat(
            reshape(
                Reactant.promote_to(
                    TracedRArray{Reactant.unwrapped_eltype(y),2},
                    LinearAlgebra.I(size(y, 1)),
                ),
                size(y, 1),
                1,
                size(y, 1),
            ),
            1,
            size(y, 2),
            1,
        ),
    )

    x = Utils.contiguous(reshape(x, :, B))

    # TODO: replace once https://github.com/LuxDL/Lux.jl/issues/1523 is fixed
    #=
    dx = similar(x, size(x, 1), size(x, 2), size(y, 1))
    fill!(dx, false)

    Enzyme.autodiff(
        ad.mode,
        Utils.annotate_enzyme_function(ad, f′),
        Reactant.StackedBatchDuplicated(y, dy),
        Reactant.StackedBatchDuplicated(x, dx),
    )

    return permutedims(dx, (3, 1, 2))
    =#

    # Our loop to batch pass should automatically batch this loop and current has better
    # coverage than the above. Though we should fix the above to ensure we never have a
    # loop in the final result.
    dx = similar(x, size(y, 1), size(x, 1), size(x, 2))
    @trace track_numbers = false for i in 1:size(y, 1)
        dxᵢ = Enzyme.make_zero(x)
        Enzyme.autodiff(
            ad.mode,
            Utils.annotate_enzyme_function(ad, f′),
            Duplicated,
            Duplicated(y, dy[:, :, i]),
            Duplicated(x, dxᵢ),
        )
        dx[i, :, :] = dxᵢ
    end
    return dx
end

function _batched_jacobian_forward_impl(f::F, ad::AutoEnzyme, x::AbstractArray) where {F}
    y, B = _check_validity_for_batched_jacobian(f, x)
    y = Utils.contiguous(reshape(y, :, B)) # will be DCEd away

    f′ = ApplyWithReshape(f, size(x))
    x = Utils.contiguous(reshape(x, :, size(x, ndims(x))))

    bx = Utils.contiguous(
        repeat(
            reshape(
                Reactant.promote_to(
                    TracedRArray{Reactant.unwrapped_eltype(x),2},
                    LinearAlgebra.I(size(x, 1)),
                ),
                size(x, 1),
                1,
                size(x, 1),
            ),
            1,
            size(x, 2),
            1,
        ),
    )

    # TODO: replace once https://github.com/LuxDL/Lux.jl/issues/1523 is fixed
    # return stack(
    #     only(
    #         Enzyme.autodiff(
    #             ad.mode,
    #             Utils.annotate_enzyme_function(ad, f′),
    #             Reactant.StackedBatchDuplicated(x, bx),
    #         ),
    #     );
    #     dims=2,
    # )

    dy = similar(y, size(y, 1), size(x, 1), size(x, 2))
    @trace track_numbers = false for i in 1:size(x, 1)
        dy[:, i, :] = only(
            Enzyme.autodiff(
                ad.mode,
                Utils.annotate_enzyme_function(ad, f′),
                Duplicated,
                Duplicated(x, bx[:, :, i]),
            ),
        )
    end
    return dy
end

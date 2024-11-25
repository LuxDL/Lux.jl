function Lux.AutoDiffInternalImpl.batched_jacobian_internal(
        f::F, ad::AutoEnzyme, x::AbstractArray, args...) where {F}
    backend = normalize_backend(True(), ad)
    return batched_enzyme_jacobian_impl(f, backend, ADTypes.mode(backend), x, args...)
end

function batched_enzyme_jacobian_impl(
        f_orig::G, ad::AutoEnzyme, ::ForwardMode, x::AbstractArray, args...) where {G}
    # We need to run the function once to get the output type. Can we use ForwardWithPrimal?
    y = f_orig(x)
    f = annotate_function(ad, f_orig)

    @argcheck y isa AbstractArray MethodError
    if ndims(y) ≤ 1 || size(y, ndims(y)) != size(x, ndims(x))
        throw(AssertionError("`batched_jacobian` only supports batched outputs \
                              (ndims(y) > 1) && size(y, ndims(y)) == size(x, ndims(x))."))
    end
    B = size(y, ndims(y))

    J = similar(x, promote_type(eltype(y), eltype(x)), prod(size(y)[1:(end - 1)]),
        prod(size(x)[1:(end - 1)]), B)

    chunk_size = min(8, length(y) ÷ B)
    partials = ntuple(_ -> zero(x), chunk_size)

    for i in 1:chunk_size:(length(x) ÷ B)
        idxs = i:min(i + chunk_size - 1, length(x) ÷ B)
        partials′ = make_onehot!(partials, idxs)
        J_partials = only(Enzyme.autodiff(
            ad.mode, f, BatchDuplicated(x, partials′), Const.(args)...))
        for (idx, J_partial) in zip(idxs, J_partials)
            copyto!(view(J, :, idx, :), reshape(J_partial, :, B))
        end
    end

    return J
end

function batched_enzyme_jacobian_impl(
        f_orig::G, ad::AutoEnzyme, ::ReverseMode, x::AbstractArray, args...) where {G}
    # We need to run the function once to get the output type. Can we use ReverseWithPrimal?
    y = f_orig(x)

    @argcheck y isa AbstractArray MethodError
    if ndims(y) ≤ 1 || size(y, ndims(y)) != size(x, ndims(x))
        throw(AssertionError("`batched_jacobian` only supports batched outputs \
                              (ndims(y) > 1) && size(y, ndims(y)) == size(x, ndims(x))."))
    end
    B = size(y, ndims(y))

    J = similar(x, promote_type(eltype(y), eltype(x)), prod(size(y)[1:(end - 1)]),
        prod(size(x)[1:(end - 1)]), B)

    chunk_size = min(8, length(x) ÷ B)
    partials = ntuple(_ -> zero(y), chunk_size)
    J_partials = ntuple(_ -> zero(x), chunk_size)

    fn = annotate_function(ad, OOPFunctionWrapper(f_orig))
    for i in 1:chunk_size:(length(y) ÷ B)
        idxs = i:min(i + chunk_size - 1, length(y) ÷ B)
        partials′ = make_onehot!(partials, idxs)
        J_partials′ = make_zero!(J_partials, idxs)
        Enzyme.autodiff(
            ad.mode, fn, BatchDuplicated(y, partials′),
            BatchDuplicated(x, J_partials′), Const.(args)...
        )
        for (idx, J_partial) in zip(idxs, J_partials)
            copyto!(view(J, idx, :, :), reshape(J_partial, :, B))
        end
    end

    return J
end

function make_onehot!(partials, idxs)
    for (idx, partial) in zip(idxs, partials)
        partial′ = reshape(partial, :, size(partial, ndims(partial)))
        fill!(partial′, false)
        fill!(view(partial′, idx, :), true)
    end
    return partials[1:length(idxs)]
end

function make_zero!(partials, idxs)
    for partial in partials
        fill!(partial, false)
    end
    return partials[1:length(idxs)]
end

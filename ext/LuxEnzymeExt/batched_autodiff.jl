function Lux.AutoDiffInternalImpl.batched_jacobian_impl(
        f::F, ad::AutoEnzyme, x::AbstractArray) where {F}
    backend = normalize_backend(True(), ad)
    return batched_enzyme_jacobian_impl(
        annotate_function(ad, f), backend, ADTypes.mode(backend), x)
end

function batched_enzyme_jacobian_impl(
        f::F, ad::AutoEnzyme, ::ForwardMode, x::AbstractArray) where {F}
    # We need to run the function once to get the output type. Can we use ForwardWithPrimal?
    y = f(x)

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
        J_partials = only(Enzyme.autodiff(ad.mode, f, BatchDuplicated(x, partials′)))
        for (idx, J_partial) in zip(idxs, J_partials)
            copyto!(view(J, :, idx, :), reshape(J_partial, :, B))
        end
    end

    return J
end

function batched_enzyme_jacobian_impl(
        f::F, ad::AutoEnzyme, ::ReverseMode, x::AbstractArray) where {F}
    error("reverse mode is not supported yet")
end

function make_onehot!(partials, idxs)
    for (idx, partial) in zip(idxs, partials)
        partial′ = reshape(partial, :, size(partial, ndims(partial)))
        fill!(partial′, false)
        partial′[idx, :] .= true
    end
    return partials[1:length(idxs)]
end

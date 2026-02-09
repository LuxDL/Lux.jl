function batched_jacobian(f::F, backend::AbstractADType, x::AbstractArray) where {F}
    return batched_jacobian_internal(f, backend, x)
end

for fType in AD_CONVERTIBLE_FUNCTIONS
    @eval function batched_jacobian(f::$(fType), backend::AbstractADType, x::AbstractArray)
        f̂, y = rewrite_autodiff_call(f)
        return batched_jacobian_internal(f̂, backend, x, y)
    end
end

function batched_jacobian_internal(
    f::F, backend::AbstractADType, x::AbstractArray, y
) where {F}
    return batched_jacobian_internal(Base.Fix2(f, y), backend, x)
end

# These are useful to extend Nested AD for non-chain rules backends
function CRC.rrule(
    ::typeof(batched_jacobian_internal), f::F, backend::AbstractADType, x::AbstractArray, y
) where {F}
    return CRC.rrule_via_ad(
        rule_config(Val(:Zygote)), batched_jacobian_internal, f, backend, x, y
    )
end

function CRC.rrule(
    cfg::RuleConfig{>:HasReverseMode},
    ::typeof(batched_jacobian_internal),
    f::F,
    backend::AbstractADType,
    x::AbstractArray,
    y,
) where {F}
    grad_fn = let cfg = cfg
        (f̂, x, args...) -> begin
            res, ∂f = CRC.rrule_via_ad(cfg, f̂, x, args...)
            return ∂f(one(res))[2:end]
        end
    end

    jac_fn = let backend = backend
        (f̂, x̃) -> batched_jacobian_internal(f̂, backend, x̃)
    end

    res, ∇autodiff_jacobian = CRC.rrule_via_ad(
        cfg, autodiff_jacobian, jac_fn, grad_fn, f, x, y
    )
    ∇batched_jacobian = let ∇autodiff_jacobian = ∇autodiff_jacobian
        Δ -> begin
            _, _, _, _, ∂x, ∂y = ∇autodiff_jacobian(tuple(Utils.recursive_unthunk(Δ)))
            return NoTangent(), NoTangent(), NoTangent(), ∂x, ∂y
        end
    end
    return res, ∇batched_jacobian
end

function CRC.rrule(
    ::typeof(batched_jacobian_internal), f::F, backend::AbstractADType, x::AbstractArray
) where {F}
    return CRC.rrule_via_ad(
        rule_config(Val(:Zygote)), batched_jacobian_internal, f, backend, x
    )
end

function CRC.rrule(
    cfg::RuleConfig{>:HasReverseMode},
    ::typeof(batched_jacobian_internal),
    f::F,
    backend::AbstractADType,
    x::AbstractArray,
) where {F}
    f̂ = let f = f
        (x, _) -> f(x)
    end

    res, ∇batched_jacobian_full = CRC.rrule_via_ad(
        cfg, batched_jacobian_internal, f̂, backend, x, nothing
    )
    ∇batched_jacobian = let ∇batched_jacobian_full = ∇batched_jacobian_full
        Δ -> begin
            ∂x = ∇batched_jacobian_full(Utils.recursive_unthunk(Δ))[4]
            return NoTangent(), NoTangent(), NoTangent(), ∂x
        end
    end
    return res, ∇batched_jacobian
end

# We need this intermediate call to ensure that there aren't any ambiguities
function batched_jacobian_internal(
    f::F, backend::AbstractADType, x::AbstractArray
) where {F}
    return batched_jacobian_impl(
        f, Lux.Training.maybe_wrap_adtype(backend, get_device_type(x)), x
    )
end

# ForwardDiff.jl Implementation
function batched_jacobian_impl(
    f::F, backend::AutoForwardDiff{CK}, x::AbstractArray
) where {F,CK}
    x_size = size(x)
    f̂ = @closure x -> f(reshape(x, x_size))
    tag = backend.tag === nothing ? ForwardDiff.Tag(f̂, eltype(x)) : backend.tag
    chunksize = if (CK === nothing || CK ≤ 0)
        ForwardDiff.Chunk{min(prod(size(x)[1:(end - 1)]), 8)}()
    else
        ForwardDiff.Chunk{CK}()
    end
    return batched_forwarddiff_jacobian(
        f̂, reshape(x, :, size(x, ndims(x))), typeof(tag), chunksize
    )
end

@views function batched_forwarddiff_jacobian(
    f::F, x::AbstractMatrix{T}, ::Type{Tag}, ck::ForwardDiff.Chunk{CK}
) where {F,T,Tag,CK}
    N, B = size(x)
    @assert CK ≤ N

    nchunks, remainder = divrem(N, CK)

    dual_type = ForwardDiff.Dual{Tag,eltype(x),CK}
    partials_type = ForwardDiff.Partials{CK,eltype(x)}

    J_partial = batched_forwarddiff_jacobian_chunk!!(
        nothing, f, x, Tag, ck, dual_type, partials_type, 1
    )

    nchunks == 1 && remainder == 0 && return J_partial

    J = similar(J_partial, size(J_partial, 1), N, B)
    J[:, 1:CK, :] .= J_partial

    Threads.@threads for i in 2:nchunks
        batched_forwarddiff_jacobian_chunk!!(
            J[:, ((i - 1) * CK + 1):(i * CK), :],
            f,
            x,
            Tag,
            ck,
            dual_type,
            partials_type,
            (i - 1) * CK + 1,
        )
    end

    # for the remainder term we could construct a new Dual and Partial but that causes
    # problems on GPU. So we just do some extra work here.
    if remainder > 0
        batched_forwarddiff_jacobian_chunk!!(
            J[:, (end - CK + 1):end, :], f, x, Tag, ck, dual_type, partials_type, N - CK + 1
        )
    end

    return J
end

@views function batched_forwarddiff_jacobian_chunk!!(
    J_partial::Union{Nothing,AbstractArray{<:Real,3}},
    f::F,
    x::AbstractMatrix{T},
    ::Type{Tag},
    ::ForwardDiff.Chunk{CK},
    ::Type{Dual},
    ::Type{Partials},
    idx::Int,
) where {F,T,Tag,CK,Dual,Partials}
    N, B = size(x)

    idxs = idx:min(idx + CK - 1, N)
    idxs_prev = 1:(idx - 1)
    idxs_next = (idx + CK):N

    dev = get_device(x)

    partials = map(
        𝒾 -> Partials(ntuple(𝒿 -> ifelse(𝒾 == 𝒿, oneunit(T), zero(T)), CK)),
        dev(collect(1:length(idxs))),
    )
    x_part_duals = Dual.(x[idxs, :], partials)

    if length(idxs_prev) == 0
        x_part_prev = similar(x_part_duals, 0, B)
    else
        x_part_prev =
            Dual.(
                x[idxs_prev, :],
                map(
                    𝒾 -> Partials(ntuple(_ -> zero(T), CK)),
                    dev(collect(1:length(idxs_prev))),
                ),
            )
    end

    if length(idxs_next) == 0
        x_part_next = similar(x_part_duals, 0, B)
    else
        x_part_next =
            Dual.(
                x[idxs_next, :],
                map(
                    𝒾 -> Partials(ntuple(_ -> zero(T), CK)),
                    dev(collect(1:length(idxs_next))),
                ),
            )
    end

    x_duals = vcat(x_part_prev, x_part_duals, x_part_next)
    y_duals_ = f(x_duals)
    @assert ndims(y_duals_) > 1 && size(y_duals_, ndims(y_duals_)) == B
    y_duals = reshape(y_duals_, :, B)

    partials_wrap(y, i) = ForwardDiff.partials(Tag, y, i)
    J_partial === nothing && return stack(i -> partials_wrap.(y_duals, i), 1:CK; dims=2)

    for i in 1:CK
        J_partial[:, i, :] .= partials_wrap.(y_duals, i)
    end
    return J_partial
end

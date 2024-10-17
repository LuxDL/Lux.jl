function extract_partials(::Type{Tag}, x, i) where {Tag}
    x isa ForwardDiff.Dual && return ForwardDiff.partials(Tag, x, i)
    if x isa AbstractArray
        bfn(xᵢ, iᵢ) = ForwardDiff.partials(Tag, xᵢ, iᵢ)
        return bfn.(x, i)
    end
    map_fn = @closure(xᵢ->extract_partials(Tag, xᵢ, i))
    (x isa Tuple || x isa NamedTuple) && return map(map_fn, x)
    x isa CRC.AbstractTangent && return extract_partials(Tag, CRC.backing(x), i)
    x === nothing && return nothing
    return fmap(map_fn, x)
end

function construct_duals(::Type{Tag}, ::Type{T}, x, u) where {Tag, T}
    if x isa AbstractArray
        bfn(xᵢ, uᵢ) = ForwardDiff.Dual{Tag, T, 1}(xᵢ, ForwardDiff.Partials{1, T}(uᵢ))
        return bfn.(x, tuple.(reshape(u, size(x))))
    end
    (x isa Tuple || x isa NamedTuple) &&
        return map((xᵢ, uᵢ) -> construct_duals(Tag, T, xᵢ, uᵢ), x, u)
    return fmap((xᵢ, uᵢ) -> construct_duals(Tag, T, xᵢ, uᵢ), x, u)
end

# This is not a general jvp code, but rather meant to be efficient for nested AD calls
function forwarddiff_jvp(f::F, x, Δx, y) where {F}
    T = promote_type(Lux.recursive_eltype(x), Lux.recursive_eltype(Δx))
    Tag = typeof(ForwardDiff.Tag(f, T))
    res1_dual, res2_dual = f(construct_duals(Tag, T, x, Δx), y)
    return (extract_partials(Tag, res1_dual, 1), extract_partials(Tag, res2_dual, 1))
end

numrows(x::AbstractMatrix) = size(x, 1)
numrows(x::AbstractArray{T, 3}) where {T} = size(x, 1) * size(x, 3)

batched_row(x::AbstractMatrix, i::Integer) = view(x, i, :)
function batched_row(x::AbstractArray{T, 3}, i::Integer) where {T}
    M, N, K = size(x)
    k = (i - 1) ÷ M + 1
    i = mod1(i, M)
    y = similar(x, N * K)
    data = view(x, i, :, k)
    fill!(view(y, 1:(N*(K-1))), zero(T))
    copyto!(view(y, (N*(k-1)+1):(N*k)), data)
    fill!(view(y, (N*k+1):(N*K)), zero(T))
    return y
end

function compactify_if_structured_matrix(
        J::AbstractArray{T1, N}, Δ::AbstractArray{T2}) where {T1, T2, N}
    @argcheck N ∈ (2, 3) "Only 2D and 3D arrays are supported for compactifying."
    if !ArrayInterface.fast_scalar_indexing(J) && ArrayInterface.isstructured(Δ)
        J_ = similar(J)
        copyto!(J_, Δ)
        return J_
    end
    return reshape(Δ, size(J))
end

function rule_config(::Val{P}) where {P}
    error("`Lux.AutoDiffInternalImpl.rule_config` for `$(P).jl` is not implemented. This \
           could be because `$(P).jl` hasn't been loaded yet. Try `using $(P)` first.")
end

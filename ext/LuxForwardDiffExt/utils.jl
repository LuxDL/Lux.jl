# Low-Level functions
@inline function Lux.__partials(::Type{Tag}, x, i) where {Tag}
    x isa ForwardDiff.Dual && return ForwardDiff.partials(Tag, x, i)
    if x isa AbstractArray
        bfn(xᵢ, iᵢ) = ForwardDiff.partials(Tag, xᵢ, iᵢ)
        return bfn.(x, i)
    end
    map_fn = @closure(xᵢ->Lux.__partials(Tag, xᵢ, i))
    x isa Tuple && return map(map_fn, x)
    x isa NamedTuple && return NamedTuple{keys(x)}(map(map_fn, values(x)))
    x isa CRC.AbstractTangent && return Lux.__partials(Tag, CRC.backing(x), i)
    x === nothing && return nothing
    return fmap(map_fn, x)
end

@inline function Lux.__dualify(::Type{Tag}, ::Type{T}, x, u) where {Tag, T}
    if x isa AbstractArray
        bfn(xᵢ, uᵢ) = ForwardDiff.Dual{Tag, T, 1}(xᵢ, ForwardDiff.Partials{1, T}(uᵢ))
        return bfn.(x, tuple.(reshape(u, size(x))))
    end
    x isa Tuple && return map((xᵢ, uᵢ) -> Lux.__dualify(Tag, T, xᵢ, uᵢ), x, u)
    x isa NamedTuple &&
        return NamedTuple{keys(x)}(map((xᵢ, uᵢ) -> Lux.__dualify(Tag, T, xᵢ, uᵢ), x, u))
    return fmap((xᵢ, uᵢ) -> Lux.__dualify(Tag, T, xᵢ, uᵢ), x, u)
end

@inline Lux.__eltype(::ForwardDiff.Dual{T, V}) where {T, V} = V
@inline Lux.__eltype(::AbstractArray{<:ForwardDiff.Dual{T, V}}) where {T, V} = V

@inline function Lux.__convert_eltype(
        ::Type{T}, x::AbstractArray{<:ForwardDiff.Dual{Tag, V, N}}) where {Tag, T, V, N}
    return ForwardDiff.Dual{Tag, T, N}.(x)
end

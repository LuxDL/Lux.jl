@inline Lux.__eltype(::TrackedArray{T}) where {T} = T
@inline Lux.__eltype(::TrackedReal{T}) where {T} = T
@inline Lux.__eltype(::AbstractArray{<:TrackedReal{T}}) where {T} = T

@inline Lux.__reverse(x::TrackedArray; dims=:) = ArrayInterface.aos_to_soa(reverse(x; dims))
@inline function Lux.__reverse(x::AbstractArray{<:TrackedReal}; dims=:)
    return ArrayInterface.aos_to_soa(reverse(x; dims))
end

# multigate: avoid soa formation
@inline function Lux._gate(x::TrackedArray{T, R, 1}, h::Int, n::Int) where {T, R}
    return x[Lux._gate(h, n)]
end
@inline function Lux._gate(x::AbstractVector{<:TrackedReal}, h::Int, n::Int)
    return ArrayInterface.aos_to_soa(view(x, Lux._gate(h, n)))
end
@inline function Lux._gate(x::TrackedArray{T, R, 2}, h::Int, n::Int) where {T, R}
    return x[Lux._gate(h, n), :]
end
@inline function Lux._gate(x::AbstractMatrix{<:TrackedReal}, h::Int, n::Int)
    return ArrayInterface.aos_to_soa(view(x, Lux._gate(h, n), :))
end

@inline function Lux.__convert_eltype(::Type{T}, x::AbstractArray{<:TrackedReal}) where {T}
    @warn "`Lux.__convert_eltype` doesn't support converting element types of ReverseDiff \
           `TrackedReal` arrays. Currently this is a no-op." maxlog=1
    return x
end

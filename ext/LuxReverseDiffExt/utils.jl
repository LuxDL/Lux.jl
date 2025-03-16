Utils.eltype(::Type{<:TrackedReal{T}}) where {T} = T

Utils.reverse(x::TrackedArray; dims=:) = ArrayInterface.aos_to_soa(reverse(x; dims))
function Utils.reverse(x::AbstractArray{<:TrackedReal}; dims=:)
    return ArrayInterface.aos_to_soa(reverse(x; dims))
end

# multigate: avoid soa formation
function Utils.gate(x::TrackedArray{T,R,1}, h::Int, n::Int) where {T,R}
    return x[Utils.gate(h, n)]
end
function Utils.gate(x::AbstractVector{<:TrackedReal}, h::Int, n::Int)
    return ArrayInterface.aos_to_soa(view(x, Utils.gate(h, n)))
end
function Utils.gate(x::TrackedArray{T,R,2}, h::Int, n::Int) where {T,R}
    return x[Utils.gate(h, n), :]
end
function Utils.gate(x::AbstractMatrix{<:TrackedReal}, h::Int, n::Int)
    return ArrayInterface.aos_to_soa(view(x, Utils.gate(h, n), :))
end

function Utils.ofeltype_array(::Type{T}, x::AbstractArray{<:TrackedReal}) where {T}
    @warn "`Utils.ofeltype_array` doesn't support converting element types of ReverseDiff \
           `TrackedReal` arrays. Currently this is a no-op." maxlog = 1
    return x
end

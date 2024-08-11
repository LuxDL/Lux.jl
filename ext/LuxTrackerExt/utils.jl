# Weight Norm Patch
@inline Lux._norm(x::TrackedArray; dims=Colon()) = sqrt.(sum(abs2.(x); dims))

# multigate chain rules
@inline Utils.gate(x::Tracker.TrackedVector, h::Int, n::Int) = x[Utils.gate(h, n)]
@inline Utils.gate(x::Tracker.TrackedMatrix, h::Int, n::Int) = x[Utils.gate(h, n), :]

function __construct_tracked_params(ps, dps)
    map_fn = (p, dp) -> Tracker.TrackedArray(Tracker.Call(), p, dp)
    return Lux.recursive_map(map_fn, ps, dps)
end

@inline Lux.__eltype(::TrackedArray{T}) where {T} = T
@inline Lux.__eltype(::TrackedReal{T}) where {T} = T
@inline Lux.__eltype(::AbstractArray{<:TrackedReal{T}}) where {T} = T

@inline Lux.__reverse(x::TrackedArray; dims=:) = ArrayInterface.aos_to_soa(reverse(x; dims))
@inline function Lux.__reverse(x::AbstractArray{<:TrackedReal}; dims=:)
    return ArrayInterface.aos_to_soa(reverse(x; dims))
end

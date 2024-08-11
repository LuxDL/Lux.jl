# Weight Norm Patch
Lux._norm(x::TrackedArray; dims=Colon()) = sqrt.(sum(abs2.(x); dims))

# multigate chain rules
Utils.gate(x::Tracker.TrackedVector, h::Int, n::Int) = x[Utils.gate(h, n)]
Utils.gate(x::Tracker.TrackedMatrix, h::Int, n::Int) = x[Utils.gate(h, n), :]

function __construct_tracked_params(ps, dps)
    map_fn = (p, dp) -> Tracker.TrackedArray(Tracker.Call(), p, dp)
    return Lux.recursive_map(map_fn, ps, dps)
end

Utils.eltype(::Type{<:TrackedReal{T}}) where {T} = T

Utils.reverse(x::TrackedArray; dims=:) = ArrayInterface.aos_to_soa(reverse(x; dims))
function Utils.reverse(x::AbstractArray{<:TrackedReal}; dims=:)
    return ArrayInterface.aos_to_soa(reverse(x; dims))
end

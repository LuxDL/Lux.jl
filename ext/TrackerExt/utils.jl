# Weight Norm Patch
Utils.norm(x::TrackedArray; dims=Colon()) = sqrt.(sum(abs2.(x); dims))

# multigate chain rules
Utils.gate(x::Tracker.TrackedVector, h::Int, n::Int) = x[Utils.gate(h, n)]
Utils.gate(x::Tracker.TrackedMatrix, h::Int, n::Int) = x[Utils.gate(h, n), :]

function construct_tracked_params(ps, dps)
    return fmap(ps, dps; exclude=isleaf) do p, dp
        Tracker.TrackedArray(Tracker.Call(), p, dp)
    end
end

Utils.eltype(::Type{<:TrackedReal{T}}) where {T} = T

Utils.reverse(x::TrackedArray; dims=:) = ArrayInterface.aos_to_soa(reverse(x; dims))
function Utils.reverse(x::AbstractArray{<:TrackedReal}; dims=:)
    return ArrayInterface.aos_to_soa(reverse(x; dims))
end

module LuxComponentArraysTrackerExt

using ComponentArrays, Tracker

function Tracker.param(ca::ComponentArray)
    x = getdata(ca)
    length(x) == 0 && return ComponentArray(Tracker.param(Float32[]), getaxes(ca))
    return ComponentArray(Tracker.param(x), getaxes(ca))
end

Tracker.extract_grad!(ca::ComponentArray) = Tracker.extract_grad!(getdata(ca))

function Base.materialize(bc::Base.Broadcast.Broadcasted{Tracker.TrackedStyle, Nothing,
    typeof(zero), <:Tuple{<:ComponentVector}})
    ca = first(bc.args)
    return ComponentArray(zero.(getdata(ca)), getaxes(ca))
end

function Base.getindex(g::Tracker.Grads, x::ComponentArray)
    Tracker.istracked(getdata(x)) || error("Object not tracked: $x")
    return g[Tracker.tracker(getdata(x))]
end

# For TrackedArrays ignore Base.maybeview
## Tracker with views doesn't work quite well
@inline function Base.getproperty(x::ComponentVector{T, <:TrackedArray},
    s::Symbol) where {T}
    return getproperty(x, Val(s))
end

@inline function Base.getproperty(x::ComponentVector{T, <:TrackedArray}, v::Val) where {T}
    return ComponentArrays._getindex(Base.getindex, x, v)
end

end

module LuxComponentArraysTrackerExt

if isdefined(Base, :get_extension)
    using ComponentArrays
    using Tracker
else
    using ..ComponentArrays
    using ..Tracker
end

Tracker.param(ca::ComponentArray) = ComponentArray(Tracker.param(getdata(ca)), getaxes(ca))

Tracker.extract_grad!(ca::ComponentArray) = Tracker.extract_grad!(getdata(ca))

function Base.materialize(bc::Base.Broadcast.Broadcasted{Tracker.TrackedStyle, Nothing,
                                                         typeof(zero),
                                                         <:Tuple{<:ComponentVector}})
    ca = first(bc.args)
    return ComponentArray(zero.(getdata(ca)), getaxes(ca))
end

function Base.getindex(g::Tracker.Grads, x::ComponentArray)
    Tracker.istracked(getdata(x)) || error("Object not tracked: $x")
    return g[Tracker.tracker(getdata(x))]
end

end

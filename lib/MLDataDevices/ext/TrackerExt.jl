module TrackerExt

using Adapt: Adapt
using MLDataDevices: Internal, AbstractDevice
using Tracker: Tracker

for op in (:get_device, :get_device_type)
    @eval Internal.$(op)(x::Tracker.TrackedArray) = Internal.$(op)(Tracker.data(x))
    @eval Internal.$(op)(x::AbstractArray{<:Tracker.TrackedReal}) =
        Internal.$(op)(Tracker.data.(x))
end

Internal.special_aos(::AbstractArray{<:Tracker.TrackedReal}) = true

function Adapt.adapt_structure(to::AbstractDevice, x::AbstractArray{<:Tracker.TrackedReal})
    @warn "AbstractArray{<:Tracker.TrackedReal} is not supported for $(to). Converting to \
           Tracker.TrackedArray." maxlog = 1
    return Adapt.adapt(to, Tracker.collect(x))
end

end

module MLDataDevicesTrackerExt

using Adapt: Adapt
using MLDataDevices: Internal, AMDGPUDevice, CUDADevice, MetalDevice, oneAPIDevice
using Tracker: Tracker

for op in (:get_device, :get_device_type)
    @eval Internal.$(op)(x::Tracker.TrackedArray) = Internal.$(op)(Tracker.data(x))
    @eval Internal.$(op)(x::AbstractArray{<:Tracker.TrackedReal}) = Internal.$(op)(Tracker.data.(x))
end

Internal.special_aos(::AbstractArray{<:Tracker.TrackedReal}) = true

for T in (AMDGPUDevice, AMDGPUDevice{Nothing}, CUDADevice,
    CUDADevice{Nothing}, MetalDevice, oneAPIDevice)
    @eval function Adapt.adapt_storage(to::$(T), x::AbstractArray{<:Tracker.TrackedReal})
        @warn "AbstractArray{<:Tracker.TrackedReal} is not supported for $(to). Converting \
               to Tracker.TrackedArray." maxlog=1
        return to(Tracker.collect(x))
    end
end

end

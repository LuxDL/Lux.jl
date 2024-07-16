module DeviceUtilsTrackerExt

using Adapt: Adapt
using DeviceUtils: DeviceUtils, AMDGPUDevice, CUDADevice, MetalDevice, oneAPIDevice
using Tracker: Tracker

for op in (:_get_device, :_get_device_type)
    @eval begin
        DeviceUtils.$op(x::Tracker.TrackedArray) = DeviceUtils.$op(Tracker.data(x))
        function DeviceUtils.$op(x::AbstractArray{<:Tracker.TrackedReal})
            return DeviceUtils.$op(Tracker.data.(x))
        end
    end
end

DeviceUtils.__special_aos(::AbstractArray{<:Tracker.TrackedReal}) = true

for T in (AMDGPUDevice, AMDGPUDevice{Nothing}, CUDADevice,
    CUDADevice{Nothing}, MetalDevice, oneAPIDevice)
    @eval function Adapt.adapt_storage(to::$(T), x::AbstractArray{<:Tracker.TrackedReal})
        @warn "AbstractArray{<:Tracker.TrackedReal} is not supported for $(to). Converting \
               to Tracker.TrackedArray." maxlog=1
        return to(Tracker.collect(x))
    end
end

end

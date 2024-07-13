module LuxDeviceUtilsTrackerExt

using Adapt: Adapt
using LuxDeviceUtils: LuxDeviceUtils, LuxAMDGPUDevice, LuxCUDADevice, LuxMetalDevice,
                      LuxoneAPIDevice
using Tracker: Tracker

for op in (:_get_device, :_get_device_type)
    @eval begin
        LuxDeviceUtils.$op(x::Tracker.TrackedArray) = LuxDeviceUtils.$op(Tracker.data(x))
        function LuxDeviceUtils.$op(x::AbstractArray{<:Tracker.TrackedReal})
            return LuxDeviceUtils.$op(Tracker.data.(x))
        end
    end
end

LuxDeviceUtils.__special_aos(::AbstractArray{<:Tracker.TrackedReal}) = true

for T in (LuxAMDGPUDevice, LuxAMDGPUDevice{Nothing}, LuxCUDADevice,
    LuxCUDADevice{Nothing}, LuxMetalDevice, LuxoneAPIDevice)
    @eval function Adapt.adapt_storage(to::$(T), x::AbstractArray{<:Tracker.TrackedReal})
        @warn "AbstractArray{<:Tracker.TrackedReal} is not supported for $(to). Converting \
               to Tracker.TrackedArray." maxlog=1
        return to(Tracker.collect(x))
    end
end

end

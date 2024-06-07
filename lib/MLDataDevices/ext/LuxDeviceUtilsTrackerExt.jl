module LuxDeviceUtilsTrackerExt

using Adapt: Adapt
using LuxDeviceUtils: LuxDeviceUtils, LuxAMDGPUDevice, LuxCUDADevice, LuxMetalDevice,
                      LuxoneAPIDevice
using Tracker: Tracker

@inline function LuxDeviceUtils.get_device(x::Tracker.TrackedArray)
    return LuxDeviceUtils.get_device(Tracker.data(x))
end
@inline function LuxDeviceUtils.get_device(x::AbstractArray{<:Tracker.TrackedReal})
    return LuxDeviceUtils.get_device(Tracker.data.(x))
end

@inline LuxDeviceUtils.__special_aos(::AbstractArray{<:Tracker.TrackedReal}) = true

for T in (LuxAMDGPUDevice, LuxAMDGPUDevice{Nothing}, LuxCUDADevice,
    LuxCUDADevice{Nothing}, LuxMetalDevice, LuxoneAPIDevice)
    @eval function Adapt.adapt_storage(to::$(T), x::AbstractArray{<:Tracker.TrackedReal})
        @warn "AbstractArray{<:Tracker.TrackedReal} is not supported for $(to). Converting \
               to Tracker.TrackedArray." maxlog=1
        return to(Tracker.collect(x))
    end
end

end

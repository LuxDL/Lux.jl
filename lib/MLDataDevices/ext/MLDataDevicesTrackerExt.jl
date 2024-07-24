module MLDataDevicesTrackerExt

using Adapt: Adapt
using MLDataDevices: MLDataDevices, AMDGPUDevice, CUDADevice, MetalDevice, oneAPIDevice
using Tracker: Tracker

for op in (:_get_device, :_get_device_type)
    @eval begin
        MLDataDevices.$op(x::Tracker.TrackedArray) = MLDataDevices.$op(Tracker.data(x))
        function MLDataDevices.$op(x::AbstractArray{<:Tracker.TrackedReal})
            return MLDataDevices.$op(Tracker.data.(x))
        end
    end
end

MLDataDevices.__special_aos(::AbstractArray{<:Tracker.TrackedReal}) = true

for T in (AMDGPUDevice, AMDGPUDevice{Nothing}, CUDADevice,
    CUDADevice{Nothing}, MetalDevice, oneAPIDevice)
    @eval function Adapt.adapt_storage(to::$(T), x::AbstractArray{<:Tracker.TrackedReal})
        @warn "AbstractArray{<:Tracker.TrackedReal} is not supported for $(to). Converting \
               to Tracker.TrackedArray." maxlog=1
        return to(Tracker.collect(x))
    end
end

end

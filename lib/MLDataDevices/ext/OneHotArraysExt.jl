module OneHotArraysExt

using Adapt: Adapt
using MLDataDevices:
    MLDataDevices, Internal, CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice
using OneHotArrays: OneHotArray

for op in (:get_device, :get_device_type)
    @eval Internal.$(op)(x::OneHotArray) = Internal.$(op)(x.indices)
end

for dev in (CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice)
    @eval function Internal.unsafe_free_internal!(D::Type{$(dev)}, x::OneHotArray)
        return Internal.unsafe_free_internal!(D, x.indices)
    end
end

end

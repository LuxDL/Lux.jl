module MLDataDevicesOneHotArraysExt

using Adapt: Adapt
using MLDataDevices: MLDataDevices, Internal, ReactantDevice, CPUDevice, CUDADevice,
                     AMDGPUDevice, MetalDevice, oneAPIDevice
using OneHotArrays: OneHotArray

for op in (:get_device, :get_device_type)
    @eval Internal.$(op)(x::OneHotArray) = Internal.$(op)(x.indices)
end

for dev in (CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice)
    @eval function Internal.unsafe_free_internal!(D::Type{$(dev)}, x::OneHotArray)
        return Internal.unsafe_free_internal!(D, x.indices)
    end
end

# Reactant doesn't pay very nicely with OneHotArrays at the moment
function Adapt.adapt_structure(dev::ReactantDevice, x::OneHotArray)
    x_cpu = Adapt.adapt_structure(CPUDevice(), x)
    return Adapt.adapt_storage(dev, convert(Array, x_cpu))
end

end

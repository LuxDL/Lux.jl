module  MLDataDevicesOneHotArraysExt

using Adapt: Adapt
using MLDataDevices: MLDataDevices, Internal, ReactantDevice, CPUDevice
using OneHotArrays: OneHotArray

for op in (:get_device, :get_device_type)
    @eval Internal.$(op)(x::OneHotArray) = Internal.$(op)(x.indices)
end

# Reactant doesn't pay very nicely with OneHotArrays at the moment
function Adapt.adapt_structure(dev::ReactantDevice, x::OneHotArray)
    x_cpu = Adapt.adapt_structure(CPUDevice(), x)
    return Adapt.adapt_storage(dev, convert(Array, x_cpu))
end

end

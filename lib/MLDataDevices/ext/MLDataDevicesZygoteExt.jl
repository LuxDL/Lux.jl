module MLDataDevicesZygoteExt

using Adapt: Adapt
using MLDataDevices: CPUDevice, AbstractDevice
using Zygote: OneElement

Adapt.adapt_structure(::CPUDevice, x::OneElement) = x
Adapt.adapt_structure(to::AbstractDevice, x::OneElement) = Adapt.adapt(to, collect(x))

end

module ZygoteExt

using Adapt: Adapt
using MLDataDevices: CPUDevice, AbstractDevice, Internal
using Zygote: OneElement

Adapt.adapt_structure(::CPUDevice, x::OneElement) = x
Adapt.adapt_structure(to::AbstractDevice, x::OneElement) = Adapt.adapt(to, collect(x))

Internal.get_device(::OneElement) = CPUDevice()
Internal.get_device_type(::OneElement) = CPUDevice

end

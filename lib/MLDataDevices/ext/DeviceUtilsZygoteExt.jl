module DeviceUtilsZygoteExt

using Adapt: Adapt
using DeviceUtils: AbstractDevice, CPUDevice
using Zygote: OneElement

Adapt.adapt_structure(::CPUDevice, x::OneElement) = x
Adapt.adapt_structure(to::AbstractDevice, x::OneElement) = Adapt.adapt(to, collect(x))

end

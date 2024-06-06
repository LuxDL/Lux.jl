module LuxDeviceUtilsZygoteExt

using Adapt: Adapt
using LuxDeviceUtils: AbstractLuxDevice, LuxCPUDevice
using Zygote: OneElement

Adapt.adapt_structure(::LuxCPUDevice, x::OneElement) = x
Adapt.adapt_structure(to::AbstractLuxDevice, x::OneElement) = Adapt.adapt(to, collect(x))

end

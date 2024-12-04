module MLDataDevicesChainRulesExt

using Adapt: Adapt
using MLDataDevices: CPUDevice, AbstractDevice
using ChainRules: OneElement

Adapt.adapt_structure(::CPUDevice, x::OneElement) = x
Adapt.adapt_structure(to::AbstractDevice, x::OneElement) = Adapt.adapt(to, collect(x))

end

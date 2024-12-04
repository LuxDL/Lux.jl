module MLDataDevicesFillArraysExt

using Adapt: Adapt
using FillArrays: AbstractFill
using MLDataDevices: CPUDevice, AbstractDevice, Internal

Adapt.adapt_structure(::CPUDevice, x::AbstractFill) = x
Adapt.adapt_structure(to::AbstractDevice, x::AbstractFill) = Adapt.adapt(to, collect(x))

Internal.get_device(::AbstractFill) = CPUDevice()
Internal.get_device_type(::AbstractFill) = CPUDevice

end

module MLDataDevicesFillArraysExt

using Adapt: Adapt
using FillArrays: AbstractFill
using MLDataDevices: CPUDevice, AbstractDevice

Adapt.adapt_structure(::CPUDevice, x::AbstractFill) = x
Adapt.adapt_structure(to::AbstractDevice, x::AbstractFill) = Adapt.adapt(to, collect(x))

end

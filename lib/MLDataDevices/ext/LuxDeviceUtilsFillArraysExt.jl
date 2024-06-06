module LuxDeviceUtilsFillArraysExt

using Adapt: Adapt
using FillArrays: FillArrays, AbstractFill
using LuxDeviceUtils: LuxDeviceUtils, LuxCPUDevice, AbstractLuxDevice

Adapt.adapt_structure(::LuxCPUDevice, x::AbstractFill) = x
Adapt.adapt_structure(to::AbstractLuxDevice, x::AbstractFill) = Adapt.adapt(to, collect(x))

end

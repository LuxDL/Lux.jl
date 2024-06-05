module LuxDeviceUtilsFillArraysExt

using Adapt: Adapt
using FillArrays: FillArrays, AbstractFill
using LuxDeviceUtils: LuxDeviceUtils, LuxCPUAdaptor, AbstractLuxDeviceAdaptor

Adapt.adapt_structure(::LuxCPUAdaptor, x::AbstractFill) = x

function Adapt.adapt_structure(to::AbstractLuxDeviceAdaptor, x::AbstractFill)
    return Adapt.adapt(to, collect(x))
end

end

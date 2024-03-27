module LuxDeviceUtilsFillArraysExt

using Adapt: Adapt
using FillArrays: FillArrays
using LuxDeviceUtils: LuxDeviceUtils, LuxCPUAdaptor

Adapt.adapt_structure(::LuxCPUAdaptor, x::FillArrays.AbstractFill) = x

function Adapt.adapt_structure(
        to::LuxDeviceUtils.AbstractLuxDeviceAdaptor, x::FillArrays.AbstractFill)
    return Adapt.adapt(to, collect(x))
end

end

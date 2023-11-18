module LuxDeviceUtilsFillArraysExt

using Adapt, FillArrays, LuxDeviceUtils

Adapt.adapt_structure(::LuxCPUAdaptor, x::FillArrays.AbstractFill) = x

function Adapt.adapt_structure(to::LuxDeviceUtils.AbstractLuxDeviceAdaptor,
        x::FillArrays.AbstractFill)
    return adapt(to, collect(x))
end

end

module LuxDeviceUtilsFillArraysExt

isdefined(Base, :get_extension) ? (using FillArrays) : (using ..FillArrays)

using Adapt, LuxDeviceUtils

Adapt.adapt_storage(::LuxCPUAdaptor, x::FillArrays.AbstractFill) = x

function Adapt.adapt_structure(to::LuxDeviceUtils.AbstractLuxDeviceAdaptor,
    x::FillArrays.AbstractFill)
    return Adapt.adapt_structure(to, collect(x))
end

end
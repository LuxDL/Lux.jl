module LuxDeviceUtilsFillArraysExt

isdefined(Base, :get_extension) ? (using FillArrays) : (using ..FillArrays)

using Adapt, LuxDeviceUtils

Adapt.adapt_storage(::LuxCPUAdaptor, x::FillArrays.AbstractFill) = x

end

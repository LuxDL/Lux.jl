module LuxDeviceUtilsLuxCUDAFillArraysExt

if isdefined(Base, :get_extension)
    using FillArrays
    using LuxCUDA
else
    using ..FillArrays
    using ..LuxCUDA
end

using Adapt, LuxDeviceUtils

Adapt.adapt_storage(::LuxCUDAAdaptor, x::FillArrays.AbstractFill) = cu(collect(x))

end

module LuxDeviceUtilsLuxAMDGPUFillArraysExt

if isdefined(Base, :get_extension)
    using FillArrays
    using LuxAMDGPU
else
    using ..FillArrays
    using ..LuxAMDGPU
end

using Adapt, LuxDeviceUtils

Adapt.adapt_storage(::LuxAMDGPUAdaptor, x::FillArrays.AbstractFill) = roc(collect(x))

end

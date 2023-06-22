module LuxLuxAMDGPUFillArraysExt

if isdefined(Base, :get_extension)
    using FillArrays
    using LuxAMDGPU
else
    using ..FillArrays
    using ..LuxAMDGPU
end

using Adapt, Lux

Adapt.adapt_storage(::Lux.LuxAMDGPUAdaptor, x::FillArrays.AbstractFill) = roc(collect(x))

end

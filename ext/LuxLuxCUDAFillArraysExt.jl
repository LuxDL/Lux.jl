module LuxLuxCUDAFillArraysExt

if isdefined(Base, :get_extension)
    using FillArrays
    using LuxCUDA
else
    using ..FillArrays
    using ..LuxCUDA
end

using Adapt, Lux

Adapt.adapt_storage(::Lux.LuxCUDAAdaptor, x::FillArrays.AbstractFill) = cu(collect(x))

end

module LuxFillArraysExt

using Adapt, Lux, FillArrays

Adapt.adapt_storage(::Lux.LuxCPUAdaptor, x::FillArrays.AbstractFill) = x

end

module LuxFillArraysExt

using Adapt, LuxCUDA, Lux, FillArrays

Adapt.adapt_storage(::Lux.LuxCUDAAdaptor, x::FillArrays.AbstractFill) = CUDA.cu(collect(x))

Adapt.adapt_storage(::Lux.LuxCPUAdaptor, x::FillArrays.AbstractFill) = x

end

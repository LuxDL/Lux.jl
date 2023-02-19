module LuxFillArraysExt

isdefined(Base, :get_extension) ? (using FillArrays) : (using ..FillArrays)

using Adapt, CUDA, Lux

Adapt.adapt_storage(::Lux.LuxCUDAAdaptor, x::FillArrays.AbstractFill) = CUDA.cu(collect(x))

Adapt.adapt_storage(::Lux.LuxCPUAdaptor, x::FillArrays.AbstractFill) = x

end

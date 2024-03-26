module LuxDeviceUtilsSparseArraysExt

using Adapt: Adapt
using LuxDeviceUtils: LuxCPUAdaptor
using SparseArrays: AbstractSparseArray

Adapt.adapt_storage(::LuxCPUAdaptor, x::AbstractSparseArray) = x

end

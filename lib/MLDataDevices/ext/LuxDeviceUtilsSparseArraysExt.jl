module LuxDeviceUtilsSparseArraysExt

using Adapt: Adapt
using LuxDeviceUtils: LuxCPUDevice
using SparseArrays: AbstractSparseArray

Adapt.adapt_storage(::LuxCPUDevice, x::AbstractSparseArray) = x

end

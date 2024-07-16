module DeviceUtilsSparseArraysExt

using Adapt: Adapt
using DeviceUtils: CPUDevice
using SparseArrays: AbstractSparseArray

Adapt.adapt_storage(::CPUDevice, x::AbstractSparseArray) = x

end

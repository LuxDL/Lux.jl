module MLDataDevicesSparseArraysExt

using Adapt: Adapt
using MLDataDevices: CPUDevice
using SparseArrays: AbstractSparseArray

Adapt.adapt_storage(::CPUDevice, x::AbstractSparseArray) = x

end

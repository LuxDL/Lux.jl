module LuxDeviceUtilsCUDASparseArraysExt

using Adapt: Adapt
using CUDA.CUSPARSE: AbstractCuSparseMatrix, AbstractCuSparseVector
using LuxDeviceUtils: LuxCPUDevice
using SparseArrays: SparseVector, SparseMatrixCSC

Adapt.adapt_storage(::LuxCPUDevice, x::AbstractCuSparseMatrix) = SparseMatrixCSC(x)
Adapt.adapt_storage(::LuxCPUDevice, x::AbstractCuSparseVector) = SparseVector(x)

end

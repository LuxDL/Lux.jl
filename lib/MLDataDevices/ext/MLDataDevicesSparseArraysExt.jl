module MLDataDevicesSparseArraysExt

using Adapt: Adapt
using MLDataDevices: CPUDevice
using SparseArrays: AbstractSparseArray

Adapt.adapt_storage(::CPUDevice{Missing}, x::AbstractSparseArray) = x
Adapt.adapt_storage(::CPUDevice{Nothing}, x::AbstractSparseArray) = x
function Adapt.adapt_storage(
    ::CPUDevice{T}, x::AbstractSparseArray
) where {T<:AbstractFloat}
    return convert(AbstractSparseArray{T}, x)
end

end

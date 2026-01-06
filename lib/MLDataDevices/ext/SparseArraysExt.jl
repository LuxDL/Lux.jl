module SparseArraysExt

using Adapt: Adapt
using MLDataDevices: CPUDevice, Internal
using SparseArrays: AbstractSparseArray, nonzeros

Internal.get_device(x::AbstractSparseArray) = Internal.get_device(nonzeros(x))
Internal.get_device_type(x::AbstractSparseArray) = Internal.get_device_type(nonzeros(x))

Adapt.adapt_storage(::CPUDevice{Missing}, x::AbstractSparseArray) = x
Adapt.adapt_storage(::CPUDevice{Nothing}, x::AbstractSparseArray) = x
function Adapt.adapt_storage(
    ::CPUDevice{T}, x::AbstractSparseArray
) where {T<:AbstractFloat}
    return convert(AbstractSparseArray{T}, x)
end

end

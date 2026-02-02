module SparseArraysExt

using Adapt: Adapt
using MLDataDevices: CPUDevice, Internal
using SparseArrays: AbstractSparseArray, nonzeros

Internal.get_device(x::AbstractSparseArray) = Internal.get_device(nonzeros(x))
Internal.get_device_type(x::AbstractSparseArray) = Internal.get_device_type(nonzeros(x))

function Adapt.adapt_storage(::CPUDevice{T}, x::AbstractSparseArray) where {T}
    if T <: AbstractFloat
        eltype(x) <: Complex && return convert(AbstractSparseArray{Complex{T}}, x)
        return convert(AbstractSparseArray{T}, x)
    end
    return x
end

end

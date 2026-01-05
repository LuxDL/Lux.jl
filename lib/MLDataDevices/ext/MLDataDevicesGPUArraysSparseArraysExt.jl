module MLDataDevicesGPUArraysSparseArraysExt

using Adapt: Adapt
using SparseArrays: SparseMatrixCSC, SparseVector
using GPUArrays: AbstractGPUSparseMatrixCSC, AbstractGPUSparseVector
using MLDataDevices: CPUDevice

for (T1, T2) in (
    (:AbstractGPUSparseMatrixCSC, :SparseMatrixCSC),
    (:AbstractGPUSparseVector, :SparseVector),
)
    @eval begin
        Adapt.adapt_storage(::CPUDevice{Missing}, x::$(T1)) = $(T2)(x)
        Adapt.adapt_storage(::CPUDevice{Nothing}, x::$(T1)) = $(T2)(x)
        Adapt.adapt_storage(::CPUDevice{T}, x::$(T1)) where {T<:AbstractFloat} = $(T2){T}(x)
    end
end

end

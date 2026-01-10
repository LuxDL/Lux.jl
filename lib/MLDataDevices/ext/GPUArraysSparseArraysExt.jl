module GPUArraysSparseArraysExt

using Adapt: Adapt
using SparseArrays: SparseMatrixCSC, SparseVector
using GPUArrays: GPUArrays, AbstractGPUSparseMatrixCSC, AbstractGPUSparseVector
using MLDataDevices: Internal, CPUDevice
using Random: Random

Adapt.adapt_storage(::CPUDevice, rng::GPUArrays.RNG) = Random.default_rng()

Internal.get_device(rng::GPUArrays.RNG) = Internal.get_device(rng.state)
Internal.get_device_type(rng::GPUArrays.RNG) = Internal.get_device_type(rng.state)

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

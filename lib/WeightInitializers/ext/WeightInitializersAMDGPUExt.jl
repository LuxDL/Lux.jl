module WeightInitializersAMDGPUExt

using AMDGPU: AMDGPU, ROCArray
using GPUArrays: RNG
using Random: Random
using WeightInitializers: DeviceAgnostic

function DeviceAgnostic.zeros(
        ::AMDGPU.rocRAND.RNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return AMDGPU.zeros(T, dims...)
end
function DeviceAgnostic.ones(
        ::AMDGPU.rocRAND.RNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return AMDGPU.ones(T, dims...)
end

function DeviceAgnostic.zeros(
        ::RNG, ::ROCArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return AMDGPU.zeros(T, dims...)
end
function DeviceAgnostic.ones(
        ::RNG, ::ROCArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return AMDGPU.ones(T, dims...)
end
function DeviceAgnostic.rand(
        rng::RNG, ::ROCArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = ROCArray{T}(undef, dims...)
    Random.rand!(rng, y)
    return y
end
function DeviceAgnostic.randn(
        rng::RNG, ::ROCArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = ROCArray{T}(undef, dims...)
    Random.randn!(rng, y)
    return y
end

end

module WeightInitializersAMDGPUExt

using AMDGPU: AMDGPU, ROCArray
using GPUArrays: RNG
using Random: Random
using WeightInitializers: WeightInitializers

@inline function WeightInitializers.__zeros(
        ::AMDGPU.rocRAND.RNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return AMDGPU.zeros(T, dims...)
end
@inline function WeightInitializers.__ones(
        ::AMDGPU.rocRAND.RNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return AMDGPU.ones(T, dims...)
end

@inline function WeightInitializers.__zeros(
        ::RNG, ::ROCArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return AMDGPU.zeros(T, dims...)
end
@inline function WeightInitializers.__ones(
        ::RNG, ::ROCArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return AMDGPU.ones(T, dims...)
end
@inline function WeightInitializers.__rand(
        rng::RNG, ::ROCArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = ROCArray{T}(undef, dims...)
    Random.rand!(rng, y)
    return y
end
@inline function WeightInitializers.__randn(
        rng::RNG, ::ROCArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = ROCArray{T}(undef, dims...)
    Random.randn!(rng, y)
    return y
end

end

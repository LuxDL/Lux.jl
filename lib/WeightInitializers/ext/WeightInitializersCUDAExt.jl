module WeightInitializersCUDAExt

using CUDA: CUDA, CURAND, CuArray
using GPUArrays: RNG
using Random: Random
using WeightInitializers: DeviceAgnostic

const AbstractCuRNG = Union{CUDA.RNG, CURAND.RNG}

function DeviceAgnostic.zeros(
        ::AbstractCuRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return CUDA.zeros(T, dims...)
end
function DeviceAgnostic.ones(
        ::AbstractCuRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return CUDA.ones(T, dims...)
end

function DeviceAgnostic.zeros(
        ::RNG, ::CuArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return CUDA.zeros(T, dims...)
end
function DeviceAgnostic.ones(
        ::RNG, ::CuArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return CUDA.ones(T, dims...)
end
function DeviceAgnostic.rand(
        rng::RNG, ::CuArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = CuArray{T}(undef, dims...)
    Random.rand!(rng, y)
    return y
end
function DeviceAgnostic.randn(
        rng::RNG, ::CuArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = CuArray{T}(undef, dims...)
    Random.randn!(rng, y)
    return y
end

end

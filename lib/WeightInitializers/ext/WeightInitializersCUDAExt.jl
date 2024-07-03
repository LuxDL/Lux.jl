module WeightInitializersCUDAExt

using CUDA: CUDA, CURAND, CuArray
using GPUArrays: RNG
using Random: Random
using WeightInitializers: WeightInitializers

const AbstractCuRNG = Union{CUDA.RNG, CURAND.RNG}

@inline function WeightInitializers.__zeros(
        ::AbstractCuRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return CUDA.zeros(T, dims...)
end
@inline function WeightInitializers.__ones(
        ::AbstractCuRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return CUDA.ones(T, dims...)
end

@inline function WeightInitializers.__zeros(
        ::RNG, ::CuArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return CUDA.zeros(T, dims...)
end
@inline function WeightInitializers.__ones(
        ::RNG, ::CuArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return CUDA.ones(T, dims...)
end
@inline function WeightInitializers.__rand(
        rng::RNG, ::CuArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = CuArray{T}(undef, dims...)
    Random.rand!(rng, y)
    return y
end
@inline function WeightInitializers.__randn(
        rng::RNG, ::CuArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = CuArray{T}(undef, dims...)
    Random.randn!(rng, y)
    return y
end

end

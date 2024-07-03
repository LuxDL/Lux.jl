module WeightInitializersoneAPIExt

using oneAPI: oneArray
using GPUArrays: RNG
using Random: Random
using WeightInitializers: WeightInitializers

@inline function WeightInitializers.__zeros(
        ::RNG, ::oneArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return oneAPI.zeros(T, dims...)
end
@inline function WeightInitializers.__ones(
        ::RNG, ::oneArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return oneAPI.ones(T, dims...)
end
@inline function WeightInitializers.__rand(
        rng::RNG, ::oneArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = oneArray{T}(undef, dims...)
    Random.rand!(rng, y)
    return y
end
@inline function WeightInitializers.__randn(
        rng::RNG, ::oneArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = oneArray{T}(undef, dims...)
    Random.randn!(rng, y)
    return y
end

end

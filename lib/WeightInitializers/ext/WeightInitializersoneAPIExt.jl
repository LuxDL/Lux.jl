module WeightInitializersoneAPIExt

using oneAPI: oneAPI, oneArray
using GPUArrays: RNG
using Random: Random
using WeightInitializers: WeightInitializers

function WeightInitializers.__zeros(
        ::RNG, ::oneArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return oneAPI.zeros(T, dims...)
end
function WeightInitializers.__ones(
        ::RNG, ::oneArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return oneAPI.ones(T, dims...)
end
function WeightInitializers.__rand(
        rng::RNG, ::oneArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = oneArray{T}(undef, dims...)
    Random.rand!(rng, y)
    return y
end
function WeightInitializers.__randn(
        rng::RNG, ::oneArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = oneArray{T}(undef, dims...)
    Random.randn!(rng, y)
    return y
end

end

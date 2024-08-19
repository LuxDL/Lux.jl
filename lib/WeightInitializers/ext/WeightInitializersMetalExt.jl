module WeightInitializersMetalExt

using Metal: Metal, MtlArray
using GPUArrays: RNG
using Random: Random
using WeightInitializers: WeightInitializers

function WeightInitializers.__zeros(
        ::RNG, ::MtlArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return Metal.zeros(T, dims...)
end
function WeightInitializers.__ones(
        ::RNG, ::MtlArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return Metal.ones(T, dims...)
end
function WeightInitializers.__rand(
        rng::RNG, ::MtlArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = MtlArray{T}(undef, dims...)
    Random.rand!(rng, y)
    return y
end
function WeightInitializers.__randn(
        rng::RNG, ::MtlArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = MtlArray{T}(undef, dims...)
    Random.randn!(rng, y)
    return y
end

end

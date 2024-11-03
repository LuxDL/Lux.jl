module WeightInitializersMetalExt

using Metal: Metal, MtlArray
using GPUArrays: RNG
using Random: Random
using WeightInitializers: DeviceAgnostic

function DeviceAgnostic.zeros(
        ::RNG, ::MtlArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return Metal.zeros(T, dims...)
end
function DeviceAgnostic.ones(
        ::RNG, ::MtlArray, ::Type{T}, dims::Integer...) where {T <: Number}
    return Metal.ones(T, dims...)
end
function DeviceAgnostic.rand(
        rng::RNG, ::MtlArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = MtlArray{T}(undef, dims...)
    Random.rand!(rng, y)
    return y
end
function DeviceAgnostic.randn(
        rng::RNG, ::MtlArray, ::Type{T}, dims::Integer...) where {T <: Number}
    y = MtlArray{T}(undef, dims...)
    Random.randn!(rng, y)
    return y
end

end

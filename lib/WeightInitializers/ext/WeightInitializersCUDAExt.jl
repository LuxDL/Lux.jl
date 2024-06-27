module WeightInitializersCUDAExt

using CUDA: CUDA, CURAND
using WeightInitializers: WeightInitializers

const AbstractCuRNG = Union{CUDA.RNG, CURAND.RNG}

function WeightInitializers.__zeros(::AbstractCuRNG, T::Type, dims::Integer...)
    return CUDA.zeros(T, dims...)
end
function WeightInitializers.__ones(::AbstractCuRNG, T::Type, dims::Integer...)
    return CUDA.ones(T, dims...)
end

end

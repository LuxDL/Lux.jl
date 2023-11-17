module CUDAExtWI

using WeightInitializers, CUDA

function WeightInitializers.zeros32(::Union{CUDA.RNG, CURAND.RNG}, dims...)
    return CUDA.zeros(Float32, dims...)
end

function WeightInitializers.ones32(::Union{CUDA.RNG, CURAND.RNG}, dims...)
    return CUDA.ones(Float32, dims...)
end

end

module WeightInitializersCUDAExt

using WeightInitializers, CUDA
import WeightInitializers: ones32, zeros32, _partial_apply

zeros32(::Union{CUDA.RNG, CURAND.RNG}, dims...) = CUDA.zeros(Float32, dims...)

ones32(::Union{CUDA.RNG, CURAND.RNG}, dims...) = CUDA.ones(Float32, dims...)

for initializer in (:ones32, :zeros32)
    @eval function ($initializer)(rng::Union{CUDA.RNG, CURAND.RNG}; kwargs...)
        return _partial_apply($initializer, (rng, (; kwargs...)))
    end
    @eval ($initializer)(; kwargs...) = _partial_apply($initializer, (; kwargs...))
end

end

module WeightInitializersCUDAExt

using WeightInitializers, CUDA
import WeightInitializers: __partial_apply, NUM_TO_FPOINT

const AbstractCuRNG = Union{CUDA.RNG, CURAND.RNG}

for T in ("16", "32", "64", "C16", "C32", "C64"), fname in (:ones, :zeros)
    name = Symbol(fname, T)
    TP = NUM_TO_FPOINT[Symbol(T)]
    @eval begin
        function WeightInitializers.$(name)(rng::AbstractCuRNG, dims::Integer...; kwargs...)
            return CUDA.$(fname)($TP, dims...; kwargs...)
        end
    end

    @eval function WeightInitializers.$(name)(rng::AbstractCuRNG; kwargs...)
        return __partial_apply($name, (rng, (; kwargs...)))
    end
end

end

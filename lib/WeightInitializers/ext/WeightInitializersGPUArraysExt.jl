module WeightInitializersGPUArraysExt

using GPUArrays: RNG
using WeightInitializers: WeightInitializers

for f in (:__zeros, :__ones, :__rand, :__randn)
    @eval @inline function WeightInitializers.$(f)(
            rng::RNG, ::Type{T}, dims::Integer...) where {T <: Number}
        return WeightInitializers.$(f)(rng, rng.state, T, dims...)
    end
end

end

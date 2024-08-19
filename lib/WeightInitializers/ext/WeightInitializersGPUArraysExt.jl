module WeightInitializersGPUArraysExt

using GPUArrays: RNG
using WeightInitializers: WeightInitializers

for f in (:__zeros, :__ones, :__rand, :__randn)
    @eval function WeightInitializers.$(f)(
            rng::RNG, ::Type{T}, dims::Integer...) where {T <: Number}
        return WeightInitializers.$(f)(rng, rng.state, T, dims...)
    end
end

## Certain backends don't support sampling Complex numbers, so we avoid hitting those
## dispatches
for f in (:__rand, :__randn)
    @eval function WeightInitializers.$(f)(
            rng::RNG, ::Type{<:Complex{T}}, args::Integer...) where {T <: Number}
        real_part = WeightInitializers.$(f)(rng, rng.state, T, args...)
        imag_part = WeightInitializers.$(f)(rng, rng.state, T, args...)
        return Complex{T}.(real_part, imag_part)
    end
end

end

module WeightInitializersGPUArraysExt

using GPUArrays: RNG
using WeightInitializers: DeviceAgnostic

for f in (:zeros, :ones, :rand, :randn)
    @eval function DeviceAgnostic.$(f)(
            rng::RNG, ::Type{T}, dims::Integer...) where {T <: Number}
        return DeviceAgnostic.$(f)(rng, rng.state, T, dims...)
    end
end

## Certain backends don't support sampling Complex numbers, so we avoid hitting those
## dispatches
for f in (:rand, :randn)
    @eval function DeviceAgnostic.$(f)(
            rng::RNG, ::Type{<:Complex{T}}, args::Integer...) where {T <: Number}
        real_part = DeviceAgnostic.$(f)(rng, rng.state, T, args...)
        imag_part = DeviceAgnostic.$(f)(rng, rng.state, T, args...)
        return Complex{T}.(real_part, imag_part)
    end
end

end

module GPUArraysExt

using GPUArrays: RNG
using WeightInitializers: DeviceAgnostic

function DeviceAgnostic.get_backend_array(rng::RNG, ::Type{T}, dims::Integer...) where {T}
    return similar(rng.state, T, dims...)
end

end

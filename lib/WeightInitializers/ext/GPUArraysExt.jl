module GPUArraysExt

using GPUArrays: RNG
using WeightInitializers: DeviceAgnostic

function DeviceAgnostic.get_backend_array(
    ::RNG{AT}, ::Type{T}, dims::Integer...
) where {AT,T}
    return similar(AT{T,length(dims)}, dims...)
end

end

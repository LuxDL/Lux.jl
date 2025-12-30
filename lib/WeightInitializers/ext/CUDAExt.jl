module CUDAExt

using CUDA: CUDA, CURAND, CuArray
using WeightInitializers: DeviceAgnostic

function DeviceAgnostic.get_backend_array(
    ::Union{CUDA.RNG,CURAND.RNG}, ::Type{T}, dims::Integer...
) where {T}
    return CuArray{T}(undef, dims...)
end

end

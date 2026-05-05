module CUDAExt

using CUDA: CUDA, CuArray
using WeightInitializers: DeviceAgnostic

const CUDA_RNG_TYPE = @static if isdefined(CUDA, :CURAND) && isdefined(CUDA.CURAND, :RNG)
    Union{CUDA.RNG,CUDA.CURAND.RNG}
else
    CUDA.RNG
end

function DeviceAgnostic.get_backend_array(
    ::CUDA_RNG_TYPE, ::Type{T}, dims::Integer...
) where {T}
    return CuArray{T}(undef, dims...)
end

end

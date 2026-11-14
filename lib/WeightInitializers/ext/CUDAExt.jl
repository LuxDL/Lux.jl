module CUDAExt

using CUDA: CUDA, CuArray
using WeightInitializers: DeviceAgnostic

const CUDA_RNG_TYPE = Union{CUDA.RNG,CUDA.cuRAND.LibraryRNG,CUDA.cuRAND.NativeRNG}

function DeviceAgnostic.get_backend_array(
    ::CUDA_RNG_TYPE, ::Type{T}, dims::Integer...
) where {T}
    return CuArray{T}(undef, dims...)
end

end

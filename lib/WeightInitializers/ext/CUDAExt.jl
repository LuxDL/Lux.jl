module CUDAExt

using CUDA: cuRAND, CuArray
using WeightInitializers: DeviceAgnostic

function DeviceAgnostic.get_backend_array(
    ::cuRAND.NativeRNG, ::Type{T}, dims::Integer...
) where {T}
    return CuArray{T}(undef, dims...)
end

end

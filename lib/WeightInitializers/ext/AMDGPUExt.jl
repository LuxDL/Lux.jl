module AMDGPUExt

using AMDGPU: AMDGPU, ROCArray
using WeightInitializers: DeviceAgnostic

function DeviceAgnostic.get_backend_array(
    ::AMDGPU.rocRAND.RNG, ::Type{T}, dims::Integer...
) where {T}
    return ROCArray{T}(undef, dims...)
end

end

@testsetup module SharedTestSetup
import Reexport: @reexport

using LuxLib, MLDataDevices
@reexport using LuxTestUtils, StableRNGs, Test, Enzyme, Zygote

LuxTestUtils.jet_target_modules!(["LuxLib"])

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
end

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
           MLDataDevices.functional(CUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
           MLDataDevices.functional(AMDGPUDevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, false))
    cuda_testing() && push!(modes, ("cuda", CuArray, true))
    amdgpu_testing() && push!(modes, ("amdgpu", ROCArray, true))
    modes
end

__generate_fixed_array(::Type{T}, sz...) where {T} = __generate_fixed_array(T, sz)
function __generate_fixed_array(::Type{T}, sz) where {T}
    return reshape(T.(collect(1:prod(sz)) ./ prod(sz)), sz...)
end
__generate_fixed_array(::Type{T}, sz::Int) where {T} = T.(collect(1:sz) ./ sz)

export MODES, StableRNG, __generate_fixed_array

end

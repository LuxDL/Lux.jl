@testsetup module SharedTestSetup
import Reexport: @reexport

using LuxLib, LuxDeviceUtils, DispatchDoctor
@reexport using LuxTestUtils, StableRNGs, Test, Zygote
import LuxTestUtils: @jet, @test_gradients, check_approx

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
           LuxDeviceUtils.functional(LuxCUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
           LuxDeviceUtils.functional(LuxAMDGPUDevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, false))
    cuda_testing() && push!(modes, ("cuda", CuArray, true))
    amdgpu_testing() && push!(modes, ("amdgpu", ROCArray, true))
    modes
end

__istraining(::Val{training}) where {training} = training

__generate_fixed_array(::Type{T}, sz...) where {T} = __generate_fixed_array(T, sz)
function __generate_fixed_array(::Type{T}, sz) where {T}
    return reshape(T.(collect(1:prod(sz)) ./ prod(sz)), sz...)
end
__generate_fixed_array(::Type{T}, sz::Int) where {T} = T.(collect(1:sz) ./ sz)

export cpu_testing, cuda_testing, amdgpu_testing, MODES, StableRNG, __istraining,
       check_approx, @jet, @test_gradients, __generate_fixed_array, allow_unstable
end

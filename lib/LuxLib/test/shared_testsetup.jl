@testsetup module SharedTestSetup
import Reexport: @reexport

using LuxLib, LuxCUDA, AMDGPU
using LuxDeviceUtils
@reexport using LuxTestUtils, StableRNGs, Test, Zygote
import LuxTestUtils: @jet, @test_gradients, check_approx

const BACKEND_GROUP = get(ENV, "BACKEND_GROUP", "All")

cpu_testing() = BACKEND_GROUP == "All" || BACKEND_GROUP == "CPU"
function cuda_testing()
    return (BACKEND_GROUP == "All" || BACKEND_GROUP == "CUDA") &&
           LuxDeviceUtils.functional(LuxCUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "All" || BACKEND_GROUP == "AMDGPU") &&
           LuxDeviceUtils.functional(LuxAMDGPUDevice)
end

const MODES = begin
    # Mode, Array Type, GPU?
    cpu_mode = ("CPU", Array, false)
    cuda_mode = ("CUDA", CuArray, true)
    amdgpu_mode = ("AMDGPU", ROCArray, true)

    modes = []
    cpu_testing() && push!(modes, cpu_mode)
    cuda_testing() && push!(modes, cuda_mode)
    amdgpu_testing() && push!(modes, amdgpu_mode)
    modes
end

get_stable_rng(seed=12345) = StableRNG(seed)

__istraining(::Val{training}) where {training} = training

@inline __generate_fixed_array(::Type{T}, sz...) where {T} = __generate_fixed_array(T, sz)
@inline function __generate_fixed_array(::Type{T}, sz) where {T}
    return reshape(T.(collect(1:prod(sz)) ./ prod(sz)), sz...)
end
@inline __generate_fixed_array(::Type{T}, sz::Int) where {T} = T.(collect(1:sz) ./ sz)

export cpu_testing, cuda_testing, amdgpu_testing, MODES, get_stable_rng, __istraining,
       check_approx, @jet, @test_gradients, __generate_fixed_array
end

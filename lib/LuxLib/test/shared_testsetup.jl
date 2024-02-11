@testsetup module SharedTestSetup
import Reexport: @reexport

using LuxLib, LuxCUDA, LuxAMDGPU
@reexport using LuxTestUtils, StableRNGs, Test, Zygote
import LuxTestUtils: @jet, @test_gradients, check_approx

const GROUP = get(ENV, "GROUP", "All")

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && LuxCUDA.functional()
amdgpu_testing() = (GROUP == "All" || GROUP == "AMDGPU") && LuxAMDGPU.functional()

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

export cpu_testing, cuda_testing, amdgpu_testing, MODES, get_stable_rng, __istraining,
       check_approx, @jet, @test_gradients
end

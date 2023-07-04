using LuxLib, LuxTestUtils, StableRNGs, Test, Zygote
using LuxCUDA
using LuxTestUtils: @jet, @test_gradients, check_approx

const GROUP = get(ENV, "GROUP", "All")

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && LuxCUDA.functional()

@static if VERSION ≥ v"1.9"
    using LuxAMDGPU
    amdgpu_testing() = (GROUP == "All" || GROUP == "AMDGPU") && LuxAMDGPU.functional()
else
    amdgpu_testing() = false
end

const MODES = begin
    # Mode, Array Type, GPU?
    cpu_mode = ("CPU", Array, false)
    cuda_mode = ("CUDA", CuArray, true)
    amdgpu_mode = @static if VERSION ≥ v"1.9"
        ("AMDGPU", ROCArray, true)
    else
        nothing
    end

    modes = []
    cpu_testing() && push!(modes, cpu_mode)
    cuda_testing() && push!(modes, cuda_mode)
    amdgpu_testing() && push!(modes, amdgpu_mode)
    modes
end

get_stable_rng(seed=12345) = StableRNG(seed)

__istraining(::Val{training}) where {training} = training

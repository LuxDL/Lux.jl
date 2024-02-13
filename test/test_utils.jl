using Lux, LuxCore, LuxLib, LuxTestUtils, Random, StableRNGs, Test, Zygote
using LuxCUDA, LuxAMDGPU
using LuxTestUtils: @jet, @test_gradients, check_approx

const GROUP = get(ENV, "GROUP", "All")

CUDA.allowscalar(false)

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && LuxCUDA.functional()
amdgpu_testing() = (GROUP == "All" || GROUP == "AMDGPU") && LuxAMDGPU.functional()

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    cpu_mode = ("CPU", Array, LuxCPUDevice(), false)
    cuda_mode = ("CUDA", CuArray, LuxCUDADevice(), true)
    amdgpu_mode = ("AMDGPU", ROCArray, LuxAMDGPUDevice(), true)

    modes = []
    cpu_testing() && push!(modes, cpu_mode)
    cuda_testing() && push!(modes, cuda_mode)
    amdgpu_testing() && push!(modes, amdgpu_mode)

    modes
end

# Some Helper Functions
function get_default_rng(mode::String)
    if mode == "CPU"
        return copy(default_device_rng(LuxCPUDevice()))
    elseif mode == "CUDA"
        return deepcopy(default_device_rng(LuxCUDADevice()))
    elseif mode == "AMDGPU"
        return deepcopy(default_device_rng(LuxAMDGPUDevice()))
    else
        error("Unknown mode: $mode")
    end
end

get_stable_rng(seed=12345) = StableRNG(seed)

# AMDGPU Specifics
function _rocRAND_functional()
    try
        default_device_rng(LuxAMDGPUDevice())
        return true
    catch
        return false
    end
end

function __display(args...)
    println()
    return display(args...)
end

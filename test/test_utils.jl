using Lux, LuxCore, LuxLib, LuxTestUtils, Test, Zygote
using LuxCUDA  # CUDA Support
using LuxTestUtils: @jet, @test_gradients, check_approx

const GROUP = get(ENV, "GROUP", "All")

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && LuxCUDA.functional()
amdgpu_testing() = (GROUP == "All" || GROUP == "AMDGPU") # && LuxAMDGPU.functional()

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    cpu_mode = ("CPU", Array, cpu, false)
    cuda_mode = ("CUDA", CuArray, gpu, true)

    modes = []
    cpu_testing() && push!(modes, cpu_mode)
    cuda_testing() && push!(modes, cuda_mode)
    modes
end

# Some Helper Functions
function get_default_rng(mode::String)
    if mode == "CPU"
        return Random.default_rng()
    elseif mode == "CUDA"
        return CUDA.RNG()
    else
        error("Unknown mode: $mode")
    end
end

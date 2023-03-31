using LuxLib, LuxTestUtils, Test, Zygote
using LuxCUDA  # CUDA Support
using LuxTestUtils: @jet, @test_gradients, check_approx

const GROUP = get(ENV, "GROUP", "All")

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && LuxCUDA.functional()
amdgpu_testing() = (GROUP == "All" || GROUP == "AMDGPU") # && LuxAMDGPU.functional()

const MODES = begin
    # Mode, Array Type, GPU?
    cpu_mode = ("CPU", Array, false)
    cuda_mode = ("CUDA", CuArray, true)

    if GROUP == "All"
        [cpu_mode, cuda_mode]
    else
        modes = []
        cpu_testing() && push!(modes, cpu_mode)
        cuda_testing() && push!(modes, cuda_mode)
        modes
    end
end

__istraining(::Val{training}) where {training} = training

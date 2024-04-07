using Lux, LuxCUDA, LuxAMDGPU

CUDA.allowscalar(false)

const GROUP = get(ENV, "GROUP", "All")

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

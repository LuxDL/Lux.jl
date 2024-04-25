using Lux, LuxCUDA, LuxAMDGPU

CUDA.allowscalar(false)

const BACKEND_GROUP = get(ENV, "BACKEND_GROUP", "All")

cpu_testing() = BACKEND_GROUP == "All" || BACKEND_GROUP == "CPU"
cuda_testing() = (BACKEND_GROUP == "All" || BACKEND_GROUP == "CUDA") && LuxCUDA.functional()
function amdgpu_testing()
    return (BACKEND_GROUP == "All" || BACKEND_GROUP == "AMDGPU") && LuxAMDGPU.functional()
end

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

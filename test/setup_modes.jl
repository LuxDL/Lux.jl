using Lux, GPUArraysCore, Pkg

GPUArraysCore.allowscalar(false)

const BACKEND_GROUP = get(ENV, "BACKEND_GROUP", "All")

if BACKEND_GROUP == "All" || BACKEND_GROUP == "CUDA"
    Pkg.add("LuxCUDA")
    using LuxCUDA
end

if BACKEND_GROUP == "All" || BACKEND_GROUP == "AMDGPU"
    Pkg.add(["AMDGPU", "LuxAMDGPU"])
    using LuxAMDGPU
end

cpu_testing() = BACKEND_GROUP == "All" || BACKEND_GROUP == "CPU"
cuda_testing() = (BACKEND_GROUP == "All" || BACKEND_GROUP == "CUDA") && LuxCUDA.functional()
function amdgpu_testing()
    return (BACKEND_GROUP == "All" || BACKEND_GROUP == "AMDGPU") && LuxAMDGPU.functional()
end

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    modes = []
    cpu_testing() && push!(modes, ("CPU", Array, LuxCPUDevice(), false))
    cuda_testing() && push!(modes, ("CUDA", CuArray, LuxCUDADevice(), true))
    amdgpu_testing() && push!(modes, ("AMDGPU", ROCArray, LuxAMDGPUDevice(), true))

    modes
end

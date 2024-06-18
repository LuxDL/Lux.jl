using Lux, LuxDeviceUtils, GPUArraysCore, Pkg

GPUArraysCore.allowscalar(false)

const BACKEND_GROUP = get(ENV, "BACKEND_GROUP", "All")

if BACKEND_GROUP == "All" || BACKEND_GROUP == "CUDA"
    using LuxCUDA
end

if BACKEND_GROUP == "All" || BACKEND_GROUP == "AMDGPU"
    using AMDGPU
end

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
    # Mode, Array Type, Device Function, GPU?
    modes = []
    cpu_testing() && push!(modes, ("CPU", Array, LuxCPUDevice(), false))
    cuda_testing() && push!(modes, ("CUDA", CuArray, LuxCUDADevice(), true))
    amdgpu_testing() && push!(modes, ("AMDGPU", ROCArray, LuxAMDGPUDevice(), true))

    modes
end

using Lux, LuxDeviceUtils, GPUArraysCore, Pkg

GPUArraysCore.allowscalar(false)

if !@isdefined(BACKEND_GROUP)
    const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
end

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
           LuxDeviceUtils.functional(LuxCUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
           LuxDeviceUtils.functional(LuxAMDGPUDevice)
end

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, LuxCPUDevice(), false))
    cuda_testing() && push!(modes, ("cuda", CuArray, LuxCUDADevice(), true))
    amdgpu_testing() && push!(modes, ("amdgpu", ROCArray, LuxAMDGPUDevice(), true))

    modes
end

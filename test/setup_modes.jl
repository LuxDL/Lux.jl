using Lux, MLDataDevices, Pkg, LuxTestUtils

if !@isdefined(BACKEND_GROUP)
    const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))
end

if !@isdefined(LUX_CURRENT_TEST_GROUP)
    const LUX_CURRENT_TEST_GROUP = lowercase(get(ENV, "LUX_CURRENT_TEST_GROUP", "all"))
end

if LuxTestUtils.test_cuda(BACKEND_GROUP) && LUX_CURRENT_TEST_GROUP != "reactant"
    using LuxCUDA
end

if LuxTestUtils.test_amdgpu(BACKEND_GROUP)
    using AMDGPU
end

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return LuxTestUtils.test_cuda(BACKEND_GROUP) && MLDataDevices.functional(CUDADevice)
end
function cuda_reactant_testing()
    return LuxTestUtils.test_cuda(BACKEND_GROUP) && LUX_CURRENT_TEST_GROUP == "reactant"
end
function amdgpu_testing()
    return LuxTestUtils.test_amdgpu(BACKEND_GROUP) && MLDataDevices.functional(AMDGPUDevice)
end

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, CPUDevice(), false))
    cuda_testing() && push!(modes, ("cuda", CuArray, CUDADevice(), true))
    cuda_reactant_testing() && push!(modes, ("cuda", Array, CPUDevice(), true))
    amdgpu_testing() && push!(modes, ("amdgpu", ROCArray, AMDGPUDevice(), true))

    modes
end

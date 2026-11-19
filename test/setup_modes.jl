using Lux, MLDataDevices, LuxTestUtils

if !@isdefined(BACKEND_GROUP)
    const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))
end

if LuxTestUtils.test_cuda(BACKEND_GROUP)
    using LuxCUDA
end

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return LuxTestUtils.test_cuda(BACKEND_GROUP) && MLDataDevices.functional(CUDADevice)
end

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, CPUDevice(), false))
    cuda_testing() && push!(modes, ("cuda", CuArray, CUDADevice(), true))

    modes
end

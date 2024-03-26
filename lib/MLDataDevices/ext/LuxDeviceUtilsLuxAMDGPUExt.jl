module LuxDeviceUtilsLuxAMDGPUExt

using LuxAMDGPU: LuxAMDGPU
using LuxDeviceUtils: LuxDeviceUtils, LuxAMDGPUDevice, reset_gpu_device!

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::Union{LuxAMDGPUDevice, <:Type{LuxAMDGPUDevice}}) = true
function LuxDeviceUtils.__is_functional(::Union{LuxAMDGPUDevice, <:Type{LuxAMDGPUDevice}})
    return LuxAMDGPU.functional()
end

end

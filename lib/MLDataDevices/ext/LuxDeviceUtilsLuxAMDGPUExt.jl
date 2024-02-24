module LuxDeviceUtilsLuxAMDGPUExt

using LuxAMDGPU, LuxDeviceUtils, Random
import Adapt: adapt_storage, adapt

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::Union{LuxAMDGPUDevice, <:Type{LuxAMDGPUDevice}}) = true
function LuxDeviceUtils.__is_functional(::Union{LuxAMDGPUDevice, <:Type{LuxAMDGPUDevice}})
    return LuxAMDGPU.functional()
end

function LuxDeviceUtils._with_device_id(::Type{LuxAMDGPUDevice}, device_id)
    id = ifelse(device_id === nothing, 0, device_id)
    old_id = AMDGPU.device_id(AMDGPU.device()) - 1
    AMDGPU.device!(AMDGPU.devices()[id + 1])
    device = LuxAMDGPUDevice(AMDGPU.device())
    AMDGPU.device!(AMDGPU.devices()[old_id + 1])
    return device
end

# Default RNG
LuxDeviceUtils.default_device_rng(::LuxAMDGPUDevice) = AMDGPU.rocrand_rng()

# Query Device from Array
LuxDeviceUtils.get_device(::AMDGPU.AnyROCArray) = LuxAMDGPUDevice()

# Device Transfer
## To GPU
adapt_storage(::LuxAMDGPUAdaptor, x) = roc(x)
adapt_storage(::LuxAMDGPUAdaptor, rng::AbstractRNG) = rng
adapt_storage(::LuxAMDGPUAdaptor, rng::Random.TaskLocalRNG) = AMDGPU.rocrand_rng()

adapt_storage(::LuxCPUAdaptor, rng::AMDGPU.rocRAND.RNG) = Random.default_rng()

end

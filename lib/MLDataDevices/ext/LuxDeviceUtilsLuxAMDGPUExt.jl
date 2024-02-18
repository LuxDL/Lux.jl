module LuxDeviceUtilsLuxAMDGPUExt

using LuxAMDGPU, LuxDeviceUtils, Random
import Adapt: adapt_storage, adapt

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::LuxAMDGPUDevice) = true
LuxDeviceUtils.__is_functional(::LuxAMDGPUDevice) = LuxAMDGPU.functional()

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

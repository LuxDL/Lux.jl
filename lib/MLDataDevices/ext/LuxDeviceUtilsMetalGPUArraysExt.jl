module LuxDeviceUtilsMetalGPUArraysExt

using GPUArrays, LuxDeviceUtils, Metal, Random
import Adapt: adapt_storage, adapt

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::LuxMetalDevice) = true
LuxDeviceUtils.__is_functional(::LuxMetalDevice) = Metal.functional()

# Default RNG
LuxDeviceUtils.default_device_rng(::LuxMetalDevice) = GPUArrays.default_rng(MtlArray)

# Query Device from Array
LuxDeviceUtils.get_device(::MtlArray) = LuxMetalDevice()

# Device Transfer
## To GPU
adapt_storage(::LuxMetalAdaptor, x) = mtl(x)
adapt_storage(::LuxMetalAdaptor, rng::AbstractRNG) = rng
adapt_storage(::LuxMetalAdaptor, rng::Random.TaskLocalRNG) = GPUArrays.default_rng(MtlArray)

end

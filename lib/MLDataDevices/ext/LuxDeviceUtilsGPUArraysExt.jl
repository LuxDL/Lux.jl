module LuxDeviceUtilsGPUArraysExt

using GPUArrays, LuxDeviceUtils, Random
import Adapt: adapt_storage, adapt

adapt_storage(::LuxCPUAdaptor, rng::GPUArrays.RNG) = Random.default_rng()

end

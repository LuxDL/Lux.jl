module LuxDeviceUtilsGPUArraysExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using LuxDeviceUtils: LuxCPUDevice
using Random: Random

Adapt.adapt_storage(::LuxCPUDevice, rng::GPUArrays.RNG) = Random.default_rng()

end

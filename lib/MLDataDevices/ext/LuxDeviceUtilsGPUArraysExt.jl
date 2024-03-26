module LuxDeviceUtilsGPUArraysExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using LuxDeviceUtils: LuxCPUAdaptor
using Random: Random

Adapt.adapt_storage(::LuxCPUAdaptor, rng::GPUArrays.RNG) = Random.default_rng()

end

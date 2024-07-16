module DeviceUtilsGPUArraysExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using DeviceUtils: CPUDevice
using Random: Random

Adapt.adapt_storage(::CPUDevice, rng::GPUArrays.RNG) = Random.default_rng()

end

module DeviceUtilsGPUArraysExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using MLDataDevices: CPUDevice
using Random: Random

Adapt.adapt_storage(::CPUDevice, rng::GPUArrays.RNG) = Random.default_rng()

end

module MLDataDevicesGPUArraysExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using MLDataDevices: Internal, CPUDevice
using Random: Random

Adapt.adapt_storage(::CPUDevice, rng::GPUArrays.RNG) = Random.default_rng()

Internal.get_device(rng::GPUArrays.RNG) = Internal.get_device(rng.state)
Internal.get_device_type(rng::GPUArrays.RNG) = Internal.get_device_type(rng.state)

end

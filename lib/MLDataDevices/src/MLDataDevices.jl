module MLDataDevices

using Adapt: Adapt
using Functors: Functors, fleaves
using Preferences: @delete_preferences!, @load_preference, @set_preferences!
using Random: AbstractRNG, Random

abstract type AbstractDevice <: Function end
abstract type AbstractGPUDevice <: AbstractDevice end

include("public.jl")
include("iterator.jl")
include("internal.jl")

export gpu_backend!, supported_gpu_backends, reset_gpu_device!
export default_device_rng
export gpu_device, cpu_device

export CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice
export get_device, get_device_type

export DeviceIterator

end

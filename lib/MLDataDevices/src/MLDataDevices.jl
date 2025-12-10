module MLDataDevices

using Adapt: Adapt
using Functors: Functors, fleaves
using Preferences: @delete_preferences!, @load_preference, @set_preferences!
using Random: AbstractRNG, Random
using SciMLPublic: @public

abstract type AbstractDevice <: Function end
abstract type AbstractCPUDevice <: AbstractDevice end
abstract type AbstractAcceleratorDevice <: AbstractDevice end
abstract type AbstractGPUDevice <: AbstractAcceleratorDevice end

include("public.jl")
include("iterator.jl")
include("internal.jl")

export gpu_backend!, supported_gpu_backends, reset_gpu_device!
export default_device_rng
export gpu_device, cpu_device
export xla_device, reactant_device

export CPUDevice
export CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice, OpenCLDevice
export XLADevice, ReactantDevice
export get_device, get_device_type

export DeviceIterator

@public isleaf

end

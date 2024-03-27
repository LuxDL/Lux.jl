module LuxDeviceUtilsMetalGPUArraysExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using LuxDeviceUtils: LuxDeviceUtils, LuxMetalAdaptor, LuxMetalDevice, reset_gpu_device!
using Metal: Metal, MtlArray
using Random: Random, AbstractRNG

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::Union{LuxMetalDevice, Type{<:LuxMetalDevice}}) = true
function LuxDeviceUtils.__is_functional(::Union{LuxMetalDevice, Type{<:LuxMetalDevice}})
    return Metal.functional()
end

# Default RNG
LuxDeviceUtils.default_device_rng(::LuxMetalDevice) = GPUArrays.default_rng(MtlArray)

# Query Device from Array
LuxDeviceUtils.get_device(::MtlArray) = LuxMetalDevice()

# Device Transfer
## To GPU
Adapt.adapt_storage(::LuxMetalAdaptor, x) = Metal.mtl(x)
Adapt.adapt_storage(::LuxMetalAdaptor, rng::AbstractRNG) = rng
function Adapt.adapt_storage(::LuxMetalAdaptor, rng::Random.TaskLocalRNG)
    return GPUArrays.default_rng(MtlArray)
end

end

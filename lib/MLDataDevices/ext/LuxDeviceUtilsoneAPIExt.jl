module LuxDeviceUtilsoneAPIExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using LuxDeviceUtils: LuxDeviceUtils, LuxoneAPIDevice, reset_gpu_device!
using oneAPI: oneAPI, oneArray

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::Union{LuxoneAPIDevice, Type{<:LuxoneAPIDevice}}) = true
function LuxDeviceUtils.__is_functional(::Union{LuxoneAPIDevice, Type{<:LuxoneAPIDevice}})
    return oneAPI.functional()
end

# Default RNG
LuxDeviceUtils.default_device_rng(::LuxoneAPIDevice) = GPUArrays.default_rng(oneArray)

# Query Device from Array
LuxDeviceUtils.get_device(::oneArray) = LuxoneAPIDevice()

# Device Transfer
## To GPU
Adapt.adapt_storage(::LuxoneAPIDevice, x) = oneArray(x)

end

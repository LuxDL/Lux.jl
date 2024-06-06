module LuxDeviceUtilsoneAPIExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using LuxDeviceUtils: LuxDeviceUtils, LuxoneAPIAdaptor, LuxoneAPIDevice, reset_gpu_device!
using oneAPI: oneAPI, oneAPIArray

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::Union{LuxoneAPIDevice, Type{<:LuxoneAPIDevice}}) = true
function LuxDeviceUtils.__is_functional(::Union{LuxoneAPIDevice, Type{<:LuxoneAPIDevice}})
    return oneAPI.functional()
end

# Default RNG
LuxDeviceUtils.default_device_rng(::LuxoneAPIDevice) = GPUArrays.default_rng(oneAPIArray)

# Query Device from Array
LuxDeviceUtils.get_device(::oneAPIArray) = LuxoneAPIDevice()

# Device Transfer
## To GPU
Adapt.adapt_storage(::LuxoneAPIAdaptor, x) = oneAPI.oneAPIArray(x)

end

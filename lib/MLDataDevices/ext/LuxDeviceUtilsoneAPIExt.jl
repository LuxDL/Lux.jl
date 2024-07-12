module LuxDeviceUtilsoneAPIExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using LuxDeviceUtils: LuxDeviceUtils, LuxoneAPIDevice, reset_gpu_device!
using oneAPI: oneAPI, oneArray, oneL0

const SUPPORTS_FP64 = Dict{oneL0.ZeDevice, Bool}()

function __init__()
    reset_gpu_device!()
    for dev in oneAPI.devices()
        SUPPORTS_FP64[dev] = oneL0.module_properties(dev).fp64flags &
                             oneL0.ZE_DEVICE_MODULE_FLAG_FP64 ==
                             oneL0.ZE_DEVICE_MODULE_FLAG_FP64
    end
end

LuxDeviceUtils.loaded(::Union{LuxoneAPIDevice, Type{<:LuxoneAPIDevice}}) = true
function LuxDeviceUtils.functional(::Union{LuxoneAPIDevice, Type{<:LuxoneAPIDevice}})
    return oneAPI.functional()
end

# Default RNG
LuxDeviceUtils.default_device_rng(::LuxoneAPIDevice) = GPUArrays.default_rng(oneArray)

# Query Device from Array
LuxDeviceUtils.get_device(::oneArray) = LuxoneAPIDevice()

LuxDeviceUtils._get_device_type(::oneArray) = LuxoneAPIDevice

# Device Transfer
## To GPU
for (T1, T2) in ((Float64, Float32), (ComplexF64, ComplexF32))
    @eval function Adapt.adapt_storage(::LuxoneAPIDevice, x::AbstractArray{$(T1)})
        if !SUPPORTS_FP64[oneAPI.device()]
            @warn LazyString(
                "Double type is not supported on this device. Using `", $(T2), "` instead.")
            return oneArray{$(T2)}(x)
        end
        return oneArray(x)
    end
end
Adapt.adapt_storage(::LuxoneAPIDevice, x::AbstractArray) = oneArray(x)

end

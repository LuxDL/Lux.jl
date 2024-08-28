module MLDataDevicesoneAPIExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using MLDataDevices: MLDataDevices, Internal, oneAPIDevice, reset_gpu_device!
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

MLDataDevices.loaded(::Union{oneAPIDevice, Type{<:oneAPIDevice}}) = true
function MLDataDevices.functional(::Union{oneAPIDevice, Type{<:oneAPIDevice}})
    return oneAPI.functional()
end

# Default RNG
MLDataDevices.default_device_rng(::oneAPIDevice) = GPUArrays.default_rng(oneArray)

# Query Device from Array
Internal.get_device(::oneArray) = oneAPIDevice()

Internal.get_device_type(::oneArray) = oneAPIDevice

# unsafe_free!
function Internal.unsafe_free_internal!(::Type{oneAPIDevice}, x::AbstractArray)
    oneAPI.unsafe_free!(x)
    return
end

# Device Transfer
## To GPU
for (T1, T2) in ((Float64, Float32), (ComplexF64, ComplexF32))
    @eval function Adapt.adapt_storage(::oneAPIDevice, x::AbstractArray{$(T1)})
        if !SUPPORTS_FP64[oneAPI.device()]
            @warn LazyString(
                "Double type is not supported on this device. Using `", $(T2), "` instead.")
            return oneArray{$(T2)}(x)
        end
        return oneArray(x)
    end
end
Adapt.adapt_storage(::oneAPIDevice, x::AbstractArray) = oneArray(x)

end

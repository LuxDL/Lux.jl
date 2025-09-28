module MLDataDevicesoneAPIExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using MLDataDevices: MLDataDevices, Internal, oneAPIDevice, reset_gpu_device!
using oneAPI: oneAPI, oneArray, oneL0

const SUPPORTS_FP64 = Dict{oneL0.ZeDevice,Bool}()

function __init__()
    reset_gpu_device!()
    for dev in oneAPI.devices()
        SUPPORTS_FP64[dev] =
            oneL0.module_properties(dev).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 ==
            oneL0.ZE_DEVICE_MODULE_FLAG_FP64
    end
    return nothing
end

MLDataDevices.loaded(::Union{oneAPIDevice,Type{<:oneAPIDevice}}) = true
function MLDataDevices.functional(::Union{oneAPIDevice,Type{<:oneAPIDevice}})
    return oneAPI.functional()
end

# Default RNG
MLDataDevices.default_device_rng(::oneAPIDevice) = GPUArrays.default_rng(oneArray)

# Query Device from Array
Internal.get_device(::oneArray) = oneAPIDevice()

Internal.get_device_type(::oneArray) = oneAPIDevice

# unsafe_free!
function Internal.unsafe_free_internal!(::Type{oneAPIDevice}, x::AbstractArray)
    if applicable(oneAPI.unsafe_free!, x)
        oneAPI.unsafe_free!(x)
    else
        @warn "oneAPI.unsafe_free! is not defined for $(typeof(x))." maxlog = 1
    end
    return nothing
end

# Device Transfer
for (T1, T2) in ((Float64, Float32), (ComplexF64, ComplexF32))
    @eval function Adapt.adapt_storage(dev::oneAPIDevice, x::AbstractArray{$(T1)})
        MLDataDevices.get_device_type(x) <: oneAPIDevice && return x

        if dev.eltype === missing
            # Default behavior: warn and convert if FP64 not supported
            if !SUPPORTS_FP64[oneAPI.device()]
                @warn LazyString(
                    "Double type is not supported on this device. Using `",
                    $(T2),
                    "` instead.",
                )
                return oneArray{$(T2)}(x)
            end
            return oneArray(x)
        elseif dev.eltype === nothing
            # Preserve eltype but check device capability
            if !SUPPORTS_FP64[oneAPI.device()] && $(T1) <: Union{Float64,ComplexF64}
                throw(
                    ArgumentError(
                        "FP64 is not supported on this device and eltype=nothing was specified",
                    ),
                )
            end
            return oneArray(x)
        else
            # Convert to specified eltype first, then move to GPU
            x_converted = MLDataDevices._maybe_convert_eltype(dev, x)
            return oneArray(x_converted)
        end
    end
end

function Adapt.adapt_storage(dev::oneAPIDevice, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: oneAPIDevice && return x

    if dev.eltype === missing || dev.eltype === nothing
        # Default behavior for non-FP64 types or when preserving type
        return oneArray(x)
    else
        # Convert to specified eltype first, then move to GPU
        x_converted = MLDataDevices._maybe_convert_eltype(dev, x)
        return oneArray(x_converted)
    end
end

end

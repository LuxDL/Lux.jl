module oneAPIExt

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
    @eval function Adapt.adapt_storage(::oneAPIDevice{Missing}, x::AbstractArray{$(T1)})
        MLDataDevices.get_device_type(x) <: oneAPIDevice && return x
        if !SUPPORTS_FP64[oneAPI.device()]
            @warn LazyString(
                "Double type is not supported on this device. Using `", $(T2), "` instead."
            )
            return oneArray{$(T2)}(x)
        end
        return oneArray(x)
    end

    @eval function Adapt.adapt_storage(::oneAPIDevice{Nothing}, x::AbstractArray{$(T1)})
        MLDataDevices.get_device_type(x) <: oneAPIDevice && return x
        if !SUPPORTS_FP64[oneAPI.device()] && $(T1) <: Union{Float64,ComplexF64}
            throw(
                ArgumentError(
                    "FP64 is not supported on this device and eltype=nothing was specified"
                ),
            )
        end
        return oneArray(x)
    end

    @eval function Adapt.adapt_storage(
        ::oneAPIDevice{T}, x::AbstractArray{$(T1)}
    ) where {T<:AbstractFloat}
        MLDataDevices.get_device_type(x) <: oneAPIDevice && eltype(x) == T && return x
        if T === Float64 && !SUPPORTS_FP64[oneAPI.device()]
            throw(ArgumentError("FP64 is not supported on this device"))
        end
        return oneArray{T}(x)
    end
end

oneapi_array_adapt(::Type{T}, x) where {T} = Internal.array_adapt(oneArray, oneArray, T, x)

function Adapt.adapt_storage(::oneAPIDevice{Missing}, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: oneAPIDevice && return x
    return oneapi_array_adapt(Missing, x)
end

function Adapt.adapt_storage(::oneAPIDevice{Nothing}, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: oneAPIDevice && return x
    return oneapi_array_adapt(Nothing, x)
end

function Adapt.adapt_storage(::oneAPIDevice{T}, x::AbstractArray) where {T<:AbstractFloat}
    MLDataDevices.get_device_type(x) <: oneAPIDevice && eltype(x) == T && return x
    if T === Float64 && !SUPPORTS_FP64[oneAPI.device()]
        throw(ArgumentError("FP64 is not supported on this device"))
    end
    return oneapi_array_adapt(T, x)
end

end

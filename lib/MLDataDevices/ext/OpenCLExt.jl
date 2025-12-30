module OpenCLExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using OpenCL: OpenCL, cl, CLArray
using MLDataDevices: MLDataDevices, Internal, OpenCLDevice, reset_gpu_device!

const SUPPORTS_FP64 = Dict{cl.Device,Bool}()

function __init__()
    reset_gpu_device!()
    for dev in vcat(cl.devices.(cl.platforms())...)
        SUPPORTS_FP64[dev] = "cl_khr_fp64" in dev.extensions
    end
    return nothing
end

MLDataDevices.loaded(::Union{OpenCLDevice,Type{<:OpenCLDevice}}) = true
function MLDataDevices.functional(::Union{OpenCLDevice,Type{<:OpenCLDevice}})
    return !isempty(cl.platforms()) && !isempty(vcat(cl.devices.(cl.platforms())...))
end

# Default RNG
MLDataDevices.default_device_rng(::OpenCLDevice) = GPUArrays.default_rng(CLArray)

# Query Device from Array
Internal.get_device(::CLArray) = OpenCLDevice()

Internal.get_device_type(::CLArray) = OpenCLDevice

# unsafe_free!
function Internal.unsafe_free_internal!(::Type{OpenCLDevice}, x::AbstractArray)
    if applicable(OpenCL.unsafe_free!, x)
        OpenCL.unsafe_free!(x)
    else
        @warn "OpenCL.unsafe_free! is not defined for $(typeof(x))." maxlog = 1
    end
    return nothing
end

# Device Transfer
for (T1, T2) in ((Float64, Float32), (ComplexF64, ComplexF32))
    @eval function Adapt.adapt_storage(::OpenCLDevice{Missing}, x::AbstractArray{$(T1)})
        MLDataDevices.get_device_type(x) <: OpenCLDevice && return x
        if !SUPPORTS_FP64[cl.device()]
            @warn LazyString(
                "Double type is not supported on this device. Using `", $(T2), "` instead."
            )
            return CLArray{$(T2)}(x)
        end
        return CLArray(x)
    end

    @eval function Adapt.adapt_storage(::OpenCLDevice{Nothing}, x::AbstractArray{$(T1)})
        MLDataDevices.get_device_type(x) <: OpenCLDevice && return x
        if !SUPPORTS_FP64[cl.device()] && $(T1) <: Union{Float64,ComplexF64}
            throw(
                ArgumentError(
                    "FP64 is not supported on this device and eltype=nothing was specified"
                ),
            )
        end
        return CLArray(x)
    end

    @eval function Adapt.adapt_storage(
        ::OpenCLDevice{T}, x::AbstractArray{$(T1)}
    ) where {T<:AbstractFloat}
        MLDataDevices.get_device_type(x) <: OpenCLDevice && eltype(x) == T && return x
        if T === Float64 && !SUPPORTS_FP64[cl.device()]
            throw(ArgumentError("FP64 is not supported on this device"))
        end
        return CLArray{T}(x)
    end
end

opencl_array_adapt(::Type{T}, x) where {T} = Internal.array_adapt(CLArray, CLArray, T, x)

function Adapt.adapt_storage(::OpenCLDevice{Missing}, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: OpenCLDevice && return x
    return opencl_array_adapt(Missing, x)
end

function Adapt.adapt_storage(::OpenCLDevice{Nothing}, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: OpenCLDevice && return x
    return opencl_array_adapt(Nothing, x)
end

function Adapt.adapt_storage(::OpenCLDevice{T}, x::AbstractArray) where {T<:AbstractFloat}
    MLDataDevices.get_device_type(x) <: OpenCLDevice && eltype(x) == T && return x
    if T === Float64 && !SUPPORTS_FP64[cl.device()]
        throw(ArgumentError("FP64 is not supported on this device"))
    end
    return opencl_array_adapt(T, x)
end

end

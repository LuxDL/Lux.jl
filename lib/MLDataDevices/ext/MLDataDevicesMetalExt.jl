module MLDataDevicesMetalExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using MLDataDevices: MLDataDevices, Internal, MetalDevice, reset_gpu_device!
using Metal: Metal, MtlArray

__init__() = reset_gpu_device!()

MLDataDevices.loaded(::Union{MetalDevice,Type{<:MetalDevice}}) = true
MLDataDevices.functional(::Union{MetalDevice,Type{<:MetalDevice}}) = Metal.functional()

# Default RNG
MLDataDevices.default_device_rng(::MetalDevice) = GPUArrays.default_rng(MtlArray)

# Query Device from Array
Internal.get_device(::MtlArray) = MetalDevice()

Internal.get_device_type(::MtlArray) = MetalDevice

# unsafe_free!
function Internal.unsafe_free_internal!(::Type{MetalDevice}, x::AbstractArray)
    if applicable(Metal.unsafe_free!, x)
        Metal.unsafe_free!(x)
    else
        @warn "Metal.unsafe_free! is not defined for $(typeof(x))." maxlog = 1
    end
    return nothing
end

# Device Transfer
function Adapt.adapt_storage(dev::MetalDevice, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: MetalDevice && return x
    if dev.eltype === missing
        # Use Metal.mtl which does automatic eltype conversion (FP64 -> FP32)
        return Metal.mtl(x)
    elseif dev.eltype === nothing
        # Preserve the original eltype
        return MtlArray(x)
    else
        # Convert to specified eltype first, then move to GPU
        x_converted = MLDataDevices._maybe_convert_eltype(dev, x)
        return MtlArray(x_converted)
    end
end

end

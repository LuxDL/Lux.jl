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
function Adapt.adapt_storage(::MetalDevice{Missing}, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: MetalDevice && return x
    return Metal.mtl(x)  # Uses Metal default conversion (FP64 -> FP32)
end

function Adapt.adapt_storage(::MetalDevice{Nothing}, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: MetalDevice && return x
    return MtlArray(x)  # Preserves eltype
end

function Adapt.adapt_storage(::MetalDevice{T}, x::AbstractArray) where {T<:AbstractFloat}
    MLDataDevices.get_device_type(x) <: MetalDevice && eltype(x) == T && return x
    
    # Convert eltype first, then move to GPU
    ET = eltype(x)
    if ET <: AbstractFloat
        return MtlArray{T}(x)
    elseif ET <: Complex{<:AbstractFloat}
        return MtlArray{Complex{T}}(x)
    else
        return MtlArray(x)  # Don't convert non-floating point types
    end
end

end

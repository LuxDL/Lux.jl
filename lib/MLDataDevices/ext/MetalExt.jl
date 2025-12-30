module MetalExt

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
metal_array_adapt(::Type{T}, x) where {T} = Internal.array_adapt(Metal.mtl, MtlArray, T, x)

function Adapt.adapt_storage(::MetalDevice{Missing}, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: MetalDevice && return x
    return metal_array_adapt(Missing, x)
end

function Adapt.adapt_storage(::MetalDevice{Nothing}, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: MetalDevice && return x
    return metal_array_adapt(Nothing, x)
end

function Adapt.adapt_storage(::MetalDevice{T}, x::AbstractArray) where {T<:AbstractFloat}
    MLDataDevices.get_device_type(x) <: MetalDevice && eltype(x) == T && return x
    return metal_array_adapt(T, x)
end

end

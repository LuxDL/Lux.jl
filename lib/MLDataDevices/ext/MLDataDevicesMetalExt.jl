module MLDataDevicesMetalExt

using Adapt: Adapt
using GPUArrays: GPUArrays
using MLDataDevices: MLDataDevices, MetalDevice, reset_gpu_device!
using Metal: Metal, MtlArray

__init__() = reset_gpu_device!()

MLDataDevices.loaded(::Union{MetalDevice, Type{<:MetalDevice}}) = true
function MLDataDevices.functional(::Union{MetalDevice, Type{<:MetalDevice}})
    return Metal.functional()
end

# Default RNG
MLDataDevices.default_device_rng(::MetalDevice) = GPUArrays.default_rng(MtlArray)

# Query Device from Array
MLDataDevices._get_device(::MtlArray) = MetalDevice()

MLDataDevices._get_device_type(::MtlArray) = MetalDevice

# Device Transfer
## To GPU
Adapt.adapt_storage(::MetalDevice, x::AbstractArray) = Metal.mtl(x)

end

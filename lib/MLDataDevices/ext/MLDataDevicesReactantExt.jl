module MLDataDevicesReactantExt

using Adapt: Adapt
using MLDataDevices: MLDataDevices, Internal, XLADevice, CPUDevice
using Reactant: Reactant, RArray

MLDataDevices.loaded(::Union{XLADevice, Type{<:XLADevice}}) = true
MLDataDevices.functional(::Union{XLADevice, Type{<:XLADevice}}) = true

# Default RNG: Forward to CPU, we will compile it
function MLDataDevices.default_device_rng(::XLADevice)
    return MLDataDevices.default_device_rng(CPUDevice())
end

# Query Device from Array
Internal.get_device(::RArray) = XLADevice()

Internal.get_device_type(::RArray) = XLADevice

# unsafe_free!
Internal.unsafe_free_internal!(::Type{XLADevice}, x::AbstractArray) = nothing

# Device Transfer
Adapt.adapt_storage(::XLADevice, x::AbstractArray) = Reactant.to_rarray(x)

end

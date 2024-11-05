module MLDataDevicesReactantExt

using Adapt: Adapt
using MLDataDevices: MLDataDevices, Internal, ReactantDevice, CPUDevice, get_device_type
using Reactant: Reactant, XLA, RArray, ConcreteRArray, ConcreteRNumber, TracedRArray,
                TracedRNumber

MLDataDevices.loaded(::Union{ReactantDevice, Type{<:ReactantDevice}}) = true
MLDataDevices.functional(::Union{ReactantDevice, Type{<:ReactantDevice}}) = true

# Default RNG: Forward to CPU, we will compile it
function MLDataDevices.default_device_rng(::ReactantDevice)
    return MLDataDevices.default_device_rng(CPUDevice())
end

# Query Device from Array
function Internal.get_device(x::Union{ConcreteRNumber, ConcreteRArray})
    client = XLA.client(x.data)
    device = XLA.device(x.data)
    return ReactantDevice(client, device)
end

function Internal.get_device(::Union{TracedRArray, TracedRNumber})
    error("`get_device` isn't meant to be called inside `Reactant.@compile` context.")
end

function Internal.get_device_type(
        ::Union{TracedRArray, TracedRNumber, ConcreteRArray, ConcreteRNumber})
    return ReactantDevice
end

# unsafe_free!
Internal.unsafe_free_internal!(::Type{ReactantDevice}, x::AbstractArray) = nothing

# Device Transfer
function Adapt.adapt_storage(
        dev::ReactantDevice, x::AbstractArray{<:Reactant.ReactantPrimitive})
    @warn "ReactantDevice got an array on device: $(get_device_type(x)). We will have to \
           transfer this via CPU." maxlog=1
    return Adapt.adapt_storage(dev, Adapt.adapt_storage(CPUDevice(), x))
end

function Adapt.adapt_storage(dev::ReactantDevice, x::Array{<:Reactant.ReactantPrimitive})
    client = dev.client === missing ? XLA.default_backend[] : dev.client
    device = dev.device === missing ? nothing : dev.device
    return ConcreteRArray(x; client, device)
end

end

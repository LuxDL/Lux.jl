module MLDataDevicesReactantExt

using Adapt: Adapt
using MLDataDevices: MLDataDevices, Internal, ReactantDevice, CPUDevice, get_device_type
using Reactant: Reactant, XLA, ConcreteRArray, ConcreteRNumber, TracedRArray,
                TracedRNumber

MLDataDevices.loaded(::Union{ReactantDevice, Type{<:ReactantDevice}}) = true
MLDataDevices.functional(::Union{ReactantDevice, Type{<:ReactantDevice}}) = true

# Default RNG
function MLDataDevices.default_device_rng(::ReactantDevice)
    return Reactant.TracedRandom.default_rng()
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
    return Adapt.adapt(dev, Adapt.adapt(CPUDevice(), x))
end

function Adapt.adapt_storage(dev::ReactantDevice, x::Array{<:Reactant.ReactantPrimitive})
    client = dev.client === missing ? XLA.default_backend[] : dev.client
    device = dev.device === missing ? nothing : dev.device
    return ConcreteRArray(x; client, device)
end

# XXX: Check for client and device and use faster implementation if possible
function Adapt.adapt_storage(dev::ReactantDevice, x::ConcreteRArray)
    dev.client === missing && dev.device === missing && return x
    @warn "Fetching `client` and `device` from a ConcreteRArray hasn't been implemented \
           yet. Using slow fallback path." maxlog=1
    return Adapt.adapt(dev, Adapt.adapt(CPUDevice(), x))
end

end

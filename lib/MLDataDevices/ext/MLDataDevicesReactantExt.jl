module MLDataDevicesReactantExt

using Adapt: Adapt
using MLDataDevices: MLDataDevices, Internal, ReactantDevice, CPUDevice
using Random: Random
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
    return ReactantDevice(XLA.client(x), XLA.device(x))
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
function Adapt.adapt_storage(dev::ReactantDevice, x::AbstractArray)
    kwargs = (;)
    dev.client === missing || (kwargs = (; kwargs..., client=dev.client))
    dev.device === missing || (kwargs = (; kwargs..., device=dev.device))
    return ConcreteRArray(x; kwargs...)
end

Adapt.adapt_storage(::CPUDevice, ::Reactant.ConcreteRNG) = Random.default_rng()

end

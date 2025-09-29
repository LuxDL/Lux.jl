module MLDataDevicesReactantExt

using Adapt: Adapt
using MLDataDevices: MLDataDevices, Internal, ReactantDevice, CPUDevice, AbstractDevice
using Random: Random
using Reactant:
    Reactant, Profiler, ConcreteRArray, ConcreteRNumber, TracedRArray, TracedRNumber

MLDataDevices.loaded(::Union{ReactantDevice,Type{<:ReactantDevice}}) = true
MLDataDevices.functional(::Union{ReactantDevice,Type{<:ReactantDevice}}) = true

# Tracing API Overloads
function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:AbstractDevice}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return T
end

function Reactant.make_tracer(
    seen, @nospecialize(prev::AbstractDevice), @nospecialize(path), mode; kwargs...
)
    return prev
end

# Call into Reactant.to_rarray
function device_to_kwargs(dev::ReactantDevice, x)
    kwargs = (;)
    dev.client === missing || (kwargs = (; kwargs..., client=dev.client))
    dev.device === missing || (kwargs = (; kwargs..., device=dev.device))
    if dev.sharding !== missing
        if dev.sharding isa IdDict
            sharding = (
                haskey(dev.sharding, x) ? dev.sharding[x] : Reactant.Sharding.NoSharding()
            )
            @assert sharding isa Reactant.Sharding.AbstractSharding
            kwargs = (; kwargs..., sharding)
        elseif dev.sharding isa Reactant.Sharding.AbstractSharding
            kwargs = (; kwargs..., dev.sharding)
        else
            throw(ArgumentError("`sharding` must be an `IdDict` or a \
                                 `Reactant.Sharding.AbstractSharding` but got \
                                 $(typeof(dev.sharding))."))
        end
    end
    return kwargs
end

function Internal.to_rarray_internal(dev::ReactantDevice, x)
    return Reactant.to_rarray(x; device_to_kwargs(dev, x)...)
end

# Default RNG
MLDataDevices.default_device_rng(::ReactantDevice) = Reactant.TracedRandom.default_rng()

# Query Device from Array
const AllConcreteTypes = Union{
    <:Reactant.ConcreteIFRTNumber,
    <:Reactant.ConcreteIFRTArray,
    <:Reactant.ConcretePJRTNumber,
    <:Reactant.ConcretePJRTArray,
}

Internal.get_device(::Type{<:AllConcreteTypes}) = ReactantDevice()
function Internal.get_device(x::AllConcreteTypes)
    return ReactantDevice(
        Reactant.XLA.client(x),
        Reactant.XLA.device(x),
        IdDict{AllConcreteTypes,Reactant.Sharding.AbstractSharding}(
            x => x.sharding.sharding
        ),
    )
end
Internal.get_device_type(::Type{<:AllConcreteTypes}) = ReactantDevice
Internal.get_device_type(::AllConcreteTypes) = ReactantDevice

function Internal.get_device(::Union{TracedRArray,TracedRNumber})
    return error(
        "`get_device` isn't meant to be called inside `Reactant.@compile` context."
    )
end
Internal.get_device_type(::Union{TracedRArray,TracedRNumber}) = ReactantDevice

# unsafe_free!
Internal.unsafe_free_internal!(::Type{ReactantDevice}, x::AbstractArray) = nothing

# Device Transfer
Profiler.@annotate "Device Transfer (Reactant)" function Adapt.adapt_storage(
    dev::ReactantDevice{C,D,S,Missing}, x::AbstractArray
) where {C,D,S}
    return ConcreteRArray(x; device_to_kwargs(dev, x)...)  # Preserves eltype
end

Profiler.@annotate "Device Transfer (Reactant)" function Adapt.adapt_storage(
    dev::ReactantDevice{C,D,S,Nothing}, x::AbstractArray
) where {C,D,S}
    return ConcreteRArray(x; device_to_kwargs(dev, x)...)  # Preserves eltype
end

Profiler.@annotate "Device Transfer (Reactant)" function Adapt.adapt_storage(
    dev::ReactantDevice{C,D,S,T}, x::AbstractArray{ET}
) where {C,D,S,T<:AbstractFloat,ET}
    # Convert eltype first, then move to device
    if ET <: AbstractFloat
        x_converted = convert(AbstractArray{T}, x)
        return ConcreteRArray(x_converted; device_to_kwargs(dev, x_converted)...)
    elseif ET <: Complex{<:AbstractFloat}
        x_converted = convert(AbstractArray{Complex{T}}, x)
        return ConcreteRArray(x_converted; device_to_kwargs(dev, x_converted)...)
    else
        return ConcreteRArray(x; device_to_kwargs(dev, x)...)
    end
end

function Adapt.adapt_storage(
    ::CPUDevice,
    T::Type{<:Union{<:Reactant.ConcretePJRTNumber,<:Reactant.ConcreteIFRTNumber}},
)
    return Reactant.unwrapped_eltype(T)
end

function Adapt.adapt_storage(
    ::CPUDevice, x::Union{<:Reactant.ConcretePJRTNumber,<:Reactant.ConcreteIFRTNumber}
)
    return Reactant.unwrapped_eltype(x)(x)
end
Adapt.adapt_storage(::CPUDevice, ::Reactant.ReactantRNG) = Random.default_rng()

end

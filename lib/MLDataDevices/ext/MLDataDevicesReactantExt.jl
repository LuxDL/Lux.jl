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

# Default RNG
MLDataDevices.default_device_rng(::ReactantDevice) = Reactant.TracedRandom.default_rng()

# Query Device from Array
@static if isdefined(Reactant, :ConcreteIFRTArray)
    const AllConcreteTypes = Union{
        Reactant.ConcreteIFRTNumber,
        Reactant.ConcreteIFRTArray,
        Reactant.ConcretePJRTNumber,
        Reactant.ConcretePJRTArray,
    }
elseif isdefined(Reactant, :ConcretePJRTArray)
    const AllConcreteTypes = Union{Reactant.ConcretePJRTNumber,Reactant.ConcretePJRTArray}
else
    const AllConcreteTypes = Union{ConcreteRNumber,ConcreteRArray}
end

function Internal.get_device(x::AllConcreteTypes)
    return ReactantDevice(
        Reactant.XLA.client(x),
        Reactant.XLA.device(x),
        IdDict{AllConcreteTypes,Reactant.Sharding.AbstractSharding}(
            x => x.sharding.sharding
        ),
    )
end
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
    dev::ReactantDevice, x::AbstractArray
)
    kwargs = (;)
    dev.client === missing || (kwargs = (; kwargs..., client=dev.client))
    dev.device === missing || (kwargs = (; kwargs..., device=dev.device))
    if dev.sharding !== missing
        if dev.sharding isa IdDict
            sharding =
                haskey(dev.sharding, x) ? dev.sharding[x] : Reactant.Sharding.NoSharding()
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
    return ConcreteRArray(x; kwargs...)
end

Adapt.adapt_storage(::CPUDevice, ::Reactant.ReactantRNG) = Random.default_rng()

end

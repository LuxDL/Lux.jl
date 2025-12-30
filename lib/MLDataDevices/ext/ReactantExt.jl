module ReactantExt

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
            if haskey(dev.sharding, x)
                sharding = dev.sharding[x]
            else
                if all(x -> x isa Reactant.Sharding.NoSharding, values(dev.sharding))
                    sharding = Reactant.Sharding.NoSharding()
                else
                    meshes = unique([
                        getfield(sharding, :mesh) for sharding in values(dev.sharding)
                    ])
                    @assert length(meshes) == 1 "Multiple meshes are not supported."
                    sharding = Reactant.Sharding.Replicated(only(meshes))
                end
            end
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

Internal.get_device(rng::Reactant.ReactantRNG) = Internal.get_device(rng.seed)
Internal.get_device_type(rng::Reactant.ReactantRNG) = Internal.get_device_type(rng.seed)

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

# Once https://github.com/EnzymeAD/Reactant.jl/pull/1770/ lands we can directly use
# `Reactant.to_rarray`
Profiler.@annotate "Device Transfer (Reactant)" function Adapt.adapt_storage(
    dev::ReactantDevice, rng::Random.TaskLocalRNG
)
    return Reactant.ReactantRNG(
        dev(Reactant.TracedRandom.make_seed(rng)), Reactant.TracedRandom.rng_algorithm(rng)
    )
end
Profiler.@annotate "Device Transfer (Reactant)" function Adapt.adapt_storage(
    dev::ReactantDevice, rng::Random.AbstractRNG
)
    return Reactant.ReactantRNG(
        dev(Reactant.TracedRandom.make_seed(rng)), Reactant.TracedRandom.rng_algorithm(rng)
    )
end
Profiler.@annotate "Device Transfer (Reactant)" function Adapt.adapt_storage(
    ::ReactantDevice, rng::Reactant.ReactantRNG
)
    return rng
end
Profiler.@annotate "Device Transfer (Reactant)" function Adapt.adapt_storage(
    dev::ReactantDevice{C,D,S,T,TN}, x::Number
) where {C,D,S,T,TN}
    typeof(x) <: TN && return ConcreteRNumber(x; device_to_kwargs(dev, x)...)
    return x
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

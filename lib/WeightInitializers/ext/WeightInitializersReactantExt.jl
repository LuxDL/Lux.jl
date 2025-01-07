module WeightInitializersReactantExt

using Reactant: Reactant, TracedUtils, TracedRNG, ConcreteRNG, TracedRArray
using WeightInitializers: DeviceAgnostic

# random numbers are automatically handled

function DeviceAgnostic.ones(::ConcreteRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return Reactant.to_rarray(ones(T, dims...))
end
function DeviceAgnostic.zeros(
        ::ConcreteRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return Reactant.to_rarray(zeros(T, dims...))
end

function DeviceAgnostic.ones(::TracedRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return TracedUtils.promote_to(TracedRArray{T, length(dims)}, ones(T, dims...))
end
function DeviceAgnostic.zeros(::TracedRNG, ::Type{T}, dims::Integer...) where {T <: Number}
    return TracedUtils.promote_to(TracedRArray{T, length(dims)}, zeros(T, dims...))
end

end

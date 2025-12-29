module ReactantExt

using Random: AbstractRNG
using Reactant: Reactant, ReactantRNG, @reactant_overlay
using Reactant.Ops: @opcall
using WeightInitializers: DeviceAgnostic

@reactant_overlay function DeviceAgnostic.get_backend_array(
    ::AbstractRNG, ::Type{T}, dims::Integer...
) where {T}
    return @opcall fill(T(0), dims)
end

function DeviceAgnostic.get_backend_array(
    rng::ReactantRNG, ::Type{T}, dims::Integer...
) where {T}
    return similar(rng.seed, T, dims)
end

end

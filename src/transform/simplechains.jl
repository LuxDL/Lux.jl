"""
    ToSimpleChainsAdaptor()

Adaptor for converting a Lux Model to SimpleChains. The returned model is still a Lux model,
and satisfies the `AbstractExplicitLayer` interfacem but all internal calculations are
performed using SimpleChains.

:::warning

There is no way to preserve trained parameters and states when converting to
`SimpleChains.jl`.

:::
"""
struct ToSimpleChainsAdaptor <: AbstractFromLuxAdaptor end

"""
    adapt(from::ToSimpleChainsAdaptor, L::AbstractExplicitLayer)

Adapt a Flux model to Lux model. See [`ToSimpleChainsAdaptor`](@ref) for more details.
"""
function Adapt.adapt(to::ToSimpleChainsAdaptor, L::AbstractExplicitLayer)
    if Base.get_extension(@__MODULE__, :LuxSimpleChainsExt) === nothing
        error("`ToSimpleChainsAdaptor` requires `SimpleChains.jl` to be loaded.")
    end
    return __to_simplechains_adaptor(L)
end

function __to_simplechains_adaptor end

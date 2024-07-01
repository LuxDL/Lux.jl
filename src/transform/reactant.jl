"""
    AutoReactant()

Compile the training loop to MLIR/XLA via `Reactant.jl`.

This has been added to Lux very recently and is under-going rapid development. Currently,
only a limited subset of Lux models can be compiled via `Reactant.jl`. If you encounter any
issues, please report them on the `Lux.jl` or `Reactant.jl` GitHub repository.
"""
struct AutoReactant end

"""
    __make_reactant_array(x)

Converts `x` to a `Reactant.ConcreteRArray` if it is not already one.
"""
function __make_reactant_array end

@inline function __make_reactant_array(nt::NamedTuple{names}) where {names}
    return NamedTuple{names}(map(__make_reactant_array, values(nt)))
end
@inline __make_reactant_array(t::Tuple) = map(__make_reactant_array, t)
@inline __make_reactant_array(x::AbstractExplicitLayer) = x

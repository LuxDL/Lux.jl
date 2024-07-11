abstract type AbstractCompilerBackend end

"""
    ReactantBackend()

Compile Lux model and gradient computation to MLIR/XLA via `Reactant.jl`.

This has been added to Lux very recently and is under-going rapid development. Currently,
only a limited subset of Lux models can be compiled via `Reactant.jl`. If you encounter any
issues, please report them on the `Lux.jl` or `Reactant.jl` GitHub repository.

See [`Lux.Experimental.single_train_step!`](@ref) or
[`Lux.Experimental.single_train_step`](@ref) for information on how to use this backend.
"""
struct ReactantBackend <: AbstractCompilerBackend end

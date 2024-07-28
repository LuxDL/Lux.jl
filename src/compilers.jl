abstract type AbstractCompilerBackend end

"""
    ReactantBackend(; input_prototype = nothing)

Compile Lux model and gradient computation to MLIR/XLA via `Reactant.jl`.

!!! tip "Newly Added Feature!"

    This has been added to Lux very recently and is under-going rapid development.
    Currently, only a limited subset of Lux models can be compiled via `Reactant.jl`. If you
    encounter any issues, please report them on the `Lux.jl` or `Reactant.jl` GitHub
    repository.

## Keyword Arguments

  - `input_prototype`: Input data representative of the data that will be used for
    inference. If this is provided, we will compile the inference function with
    `Reactant.jl` on the first call to [`Lux.Experimental.single_train_step!`](@ref) or
    [`Lux.Experimental.single_train_step`](@ref). If this is not provided, we will have to
    recompile the inference function on every call to `(::TrainState)(data)` and this will
    be prohibitively expensive.

See [`Lux.Experimental.single_train_step!`](@ref) or
[`Lux.Experimental.single_train_step`](@ref) for information on how to use this backend.
"""
@kwdef @concrete struct ReactantBackend <: AbstractCompilerBackend
    input_prototype = nothing
end

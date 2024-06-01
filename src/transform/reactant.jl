@concrete struct ToReactantAdaptor{FST, R <: AbstractRNG} <: AbstractFromLuxAdaptor
    input_prototype

    ps_transform
    rng::R

    force_allow_mixed_eltypes::Bool
    skip_compile_vjp::Bool
    force_compile_vjp::Bool
    skip_compile_jvp::Bool
    force_compile_jvp::Bool
end

function ToReactantAdaptor{FST}(input_prototype; rng=Xoshiro(123), ps_transform=identity,
        force_allow_mixed_eltypes::Bool=false, force_compile_vjp::Bool=false,
        skip_compile_vjp::Bool=false, force_compile_jvp::Bool=false,
        skip_compile_jvp::Bool=true) where {FST}
    skip_compile_vjp && @argcheck !force_compile_vjp
    skip_compile_jvp && @argcheck !force_compile_jvp

    return ToReactantAdaptor{FST}(input_prototype, ps_transform, rng,
        force_allow_mixed_eltypes, skip_compile_vjp, force_compile_vjp,
        skip_compile_jvp, force_compile_jvp)
end
function ToReactantAdaptor(args...; fixed_state_type::Val=Val(true), kwargs...)
    return ToReactantAdaptor{__unwrap_val(fixed_state_type)}(args...; kwargs...)
end

function Adapt.adapt(to::ToReactantAdaptor, model::AbstractExplicitLayer)
    if Base.get_extension(@__MODULE__, :LuxReactantExt) === nothing
        error("`ToReactantAdaptor` requires `LuxReactantExt.jl` to be loaded.")
    end
    return __to_reactant_adaptor(to, model)
end

function __to_reactant_adaptor end
function __apply_reactant end

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

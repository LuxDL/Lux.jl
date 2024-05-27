@concrete struct ToReactantAdaptor{FST, R <: AbstractRNG} <: AbstractFromLuxAdaptor
    input_prototype
    ps_transform
    rng::R
    force_compile_backward::Bool
    force_allow_mixed_eltypes::Bool
end

function ToReactantAdaptor{FST}(input_prototype; rng=Xoshiro(123), ps_transform=identity,
        force_compile_backward::Bool=false,
        force_allow_mixed_eltypes::Bool=false) where {FST}
    return ToReactantAdaptor{FST}(input_prototype, ps_transform, rng,
        force_compile_backward, force_allow_mixed_eltypes)
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

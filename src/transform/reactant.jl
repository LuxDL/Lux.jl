@concrete struct ToReactantAdaptor{FST} <: AbstractFromLuxAdaptor
    input_prototype
    force_compile_backward::Bool
end

function ToReactantAdaptor{FST}(
        input_prototype; force_compile_backward::Bool=false) where {FST}
    return ToReactantAdaptor{FST}(input_prototype, force_compile_backward)
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

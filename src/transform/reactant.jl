# TODO: Add options to compile the gradients directly using Enzyme.jl
@concrete struct ToReactantAdaptor <: AbstractFromLuxAdaptor
    input_prototype
end

function Adapt.adapt(to::ToReactantAdaptor, model::AbstractExplicitLayer)
    if Base.get_extension(@__MODULE__, :LuxReactantExt) === nothing
        error("`ToReactantAdaptor` requires `LuxReactantExt.jl` to be loaded.")
    end
    return __to_reactant_adaptor(model, to.input_prototype)
end

function __to_reactant_adaptor end

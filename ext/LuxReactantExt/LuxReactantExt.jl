module LuxReactantExt

using Enzyme: Enzyme, Const, Duplicated, Active
using Optimisers: Optimisers
using Reactant: Reactant, @compile, AnyTracedRArray, TracedRArray, TracedRNumber
using Setfield: @set!
using Static: False

using Lux: Lux, LuxOps, Training, Utils
using Lux.Training: TrainingBackendCache, ReactantBackend

Lux.is_extension_loaded(::Val{:Reactant}) = true

Utils.to_rarray(x; kwargs...) = Reactant.to_rarray(x; kwargs...)

function Utils.promote_to(::Type{T}, x::Number) where {T <: Number}
    x isa Reactant.TracedType && return x
    return Reactant.ConcreteRNumber{T}(x)
end

include("patches.jl")
include("training.jl")

end

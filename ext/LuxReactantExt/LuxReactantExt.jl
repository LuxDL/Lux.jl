module LuxReactantExt

using Enzyme: Enzyme, Const, Duplicated, Active
using Optimisers: Optimisers
using Reactant: Reactant, @compile, @code_hlo, AnyTracedRArray, TracedRArray, TracedRNumber,
                @trace
using Setfield: @set!
using Static: False

using Lux: Lux, LuxOps, Training, Utils, Recurrence
using Lux.Training: TrainingBackendCache, ReactantBackend
using LuxCore: LuxCore
using MLDataDevices: ReactantDevice

Lux.is_extension_loaded(::Val{:Reactant}) = true

Utils.to_rarray(x; kwargs...) = Reactant.to_rarray(x; kwargs...)

Utils.contiguous(x::AnyTracedRArray) = Reactant.TracedUtils.materialize_traced_array(x)

Utils.eltype(::Type{<:TracedRArray{T, N}}) where {T, N} = T
Utils.eltype(::Type{<:TracedRNumber{T}}) where {T} = T

function Utils.promote_to(::Type{T}, x::Number) where {T <: Number}
    x isa Reactant.TracedType && return x
    return Reactant.ConcreteRNumber{T}(x)
end

include("patches.jl")
include("training.jl")
include("layers.jl")

end

module LuxReactantExt

using Enzyme: Enzyme, Const
using Optimisers: Optimisers
using Reactant:
    Reactant,
    Profiler,
    @compile,
    @code_hlo,
    @jit,
    AnyTracedRArray,
    TracedRArray,
    TracedRNumber,
    PrecisionConfig
using ReactantCore: ReactantCore, @trace
using Setfield: @set!
using Static: True, False

using Lux: Lux, LuxOps, Training, Utils, StatefulLuxLayer
using Lux.Training: TrainingBackendCache, ReactantBackend
using LuxCore: LuxCore, AbstractLuxLayer
using MLDataDevices: ReactantDevice, get_device

Lux.is_extension_loaded(::Val{:Reactant}) = true

Utils.to_rarray(x; kwargs...) = Reactant.to_rarray(x; kwargs...)

Utils.contiguous(x::AnyTracedRArray) = ReactantCore.materialize_traced_array(x)

Utils.eltype(::Type{<:TracedRArray{T,N}}) where {T,N} = T
Utils.eltype(::Type{<:TracedRNumber{T}}) where {T} = T
Utils.eltype(x::Reactant.AnyTracedRArray) = Reactant.unwrapped_eltype(x)

function Utils.promote_to(::Type{T}, x::Number) where {T<:Number}
    x isa Reactant.TracedType && return x
    return Reactant.ConcreteRNumber{T}(x)
end

# For CUDA use `PrecisionConfig.HIGH`. For other backends use `PrecisionConfig.DEFAULT`.
function default_precision_config(ps)
    rdev = get_device(ps)
    @assert rdev isa ReactantDevice "Only Reactant devices are supported."
    device = rdev.device === missing ? Reactant.XLA.default_device() : rdev.device
    device_kind = string(device)
    contains(device_kind, "CUDA") && return PrecisionConfig.HIGH
    return PrecisionConfig.DEFAULT
end

include("patches.jl")
include("training.jl")
include("layers.jl")
include("tracing.jl")
include("saved_model.jl")

end

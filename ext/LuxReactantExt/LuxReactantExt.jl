module LuxReactantExt

using Enzyme: Enzyme, Const
using Preferences: load_preference
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
    precision_config_preference = lowercase(
        load_preference(Lux, "precision_config", "auto")
    )

    if precision_config_preference == "auto"
        rdev = get_device(ps)
        rdev isa ReactantDevice || return PrecisionConfig.DEFAULT
        device = rdev.device === missing ? Reactant.XLA.default_device() : rdev.device
        device_kind = string(device)
        contains(device_kind, "CUDA") && return PrecisionConfig.HIGH
        return PrecisionConfig.DEFAULT
    end

    precision_config_preference == "default" && return PrecisionConfig.DEFAULT
    precision_config_preference == "high" && return PrecisionConfig.HIGH
    precision_config_preference == "highest" && return PrecisionConfig.HIGHEST

    throw(ArgumentError("Invalid value for `precision_config` preference \
                         ($precision_config_preference). Valid choices are \"auto\", \
                         \"default\", \"high\", and \"highest\"."))
end

function with_default_precision_config(f::F, ps) where {F}
    precision_config = default_precision_config(ps)
    return Reactant.with_config(
        f; dot_general_precision=precision_config, convolution_precision=precision_config
    )
end

include("patches.jl")
include("training.jl")
include("layers.jl")
include("tracing.jl")
include("saved_model.jl")

end

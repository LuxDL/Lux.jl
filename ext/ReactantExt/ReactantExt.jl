module ReactantExt

using ADTypes: ADTypes, AutoEnzyme
using Enzyme: Enzyme, Active, Const, Duplicated
using EnzymeCore: EnzymeCore
using LinearAlgebra: LinearAlgebra
using Functors: Functors
using Preferences: load_preference
using Random: Random
using Optimisers: Optimisers
using Reactant:
    Reactant, Profiler, AnyTracedRArray, TracedRArray, TracedRNumber, PrecisionConfig
using Reactant: @compile, @code_hlo, @jit, @opcall
using ReactantCore: ReactantCore, @trace
using Setfield: @set!
using Static: True, False

using Lux: Lux, LuxOps, Training, Utils, StatefulLuxLayer
using Lux.Training: TrainingBackendCache, ReactantBackend
using Lux: get_time_dimension, time_dimension_size, init_recurrent_state
using LuxCore: LuxCore, AbstractLuxLayer
using LuxLib: LuxLib
using MLDataDevices: MLDataDevices, ReactantDevice, reactant_device, get_device

Lux.is_extension_loaded(::Val{:Reactant}) = true

Utils.contiguous(x::AnyTracedRArray) = ReactantCore.materialize_traced_array(x)

Utils.eltype(::Type{<:TracedRArray{T,N}}) where {T,N} = T
Utils.eltype(::Type{<:TracedRNumber{T}}) where {T} = T
Utils.eltype(x::Reactant.AnyTracedRArray) = Reactant.unwrapped_eltype(x)

function default_precision_config(ps)
    precision_config_preference = lowercase(
        load_preference(Lux, "precision_config", "auto")
    )

    precision_config_preference == "auto" && return PrecisionConfig.DEFAULT
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
include("batched_jacobian.jl")

# include("precompile_workloads.jl")

end

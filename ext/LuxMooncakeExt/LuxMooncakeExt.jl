module LuxMooncakeExt

using ADTypes: AutoMooncake
using Mooncake: Mooncake, value_and_pullback!!, prepare_pullback_cache, @zero_adjoint, DefaultCtx
using Setfield: @set!
using Static: False, True

using Lux: Lux, Utils, LuxLib
using Lux.Training: TrainingBackendCache, TrainState

get_config(::AutoMooncake{Nothing}) = Mooncake.Config()
get_config(backend::AutoMooncake{<:Mooncake.Config}) = backend.config

include("training.jl")

@zero_adjoint DefaultCtx Tuple{typeof(LuxLib.Impl.generate_dropout_mask), AbstractRNG, Any, Any, Any, Any}

end

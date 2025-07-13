module LuxMooncakeExt

using ADTypes: AutoMooncake
using Mooncake: Mooncake, value_and_pullback!!, prepare_pullback_cache
using Setfield: @set!
using Static: False, True

using Lux: Lux, Utils
using Lux.Training: TrainingBackendCache, TrainState

get_config(::AutoMooncake{Nothing}) = Mooncake.Config()
get_config(backend::AutoMooncake{<:Mooncake.Config}) = backend.config

include("training.jl")

end

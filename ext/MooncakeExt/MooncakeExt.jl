module MooncakeExt

using ADTypes: AutoMooncake
using Mooncake: Mooncake, value_and_pullback!!, prepare_pullback_cache
using Setfield: @set!
using Static: True

using Lux: Lux
using Lux.Training: TrainingBackendCache, TrainState

get_config(::AutoMooncake{Nothing}) = Mooncake.Config()
get_config(backend::AutoMooncake{<:Mooncake.Config}) = backend.config

include("training.jl")

end

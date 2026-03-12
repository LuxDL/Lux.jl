module MooncakeExt

using ADTypes: AutoMooncake
using Mooncake: Mooncake, value_and_pullback!!, prepare_pullback_cache
using Setfield: @set!
using Static: True

# For handling activation function switching within Lux (SLEEFPirates speedup)
# Mooncake must be able to see wrapping used for the rules themselves (see Mooncake extension for LuxLibSLEEFPirates)
# and also in LuxLib's own extension file SLEEFPiratesExt.jl
using SLEEFPirates

using Lux: Lux
using Lux.Training: TrainingBackendCache, TrainState

get_config(::AutoMooncake{Nothing}) = Mooncake.Config()
get_config(backend::AutoMooncake{<:Mooncake.Config}) = backend.config

include("training.jl")

end

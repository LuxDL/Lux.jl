module Experimental

using ..Lux, LuxDeviceUtils, Random
import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer

include("map.jl")
include("training.jl")
include("freeze.jl")
include("share_parameters.jl")

end

# Deprecations for v0.6
module Training

using ..Experimental, Reexport
@reexport using ADTypes

for f in (:TrainState, :apply_gradients, :compute_gradients)
    msg = "`Lux.Training.$(f)` has been deprecated in favor of  `Lux.Experimental.Training.$(f)`"
    @eval begin
        function $(f)(args...; kwargs...)
            Base.depwarn($(msg), $(f))
            return Experimental.Training.$(f)(args...; kwargs...)
        end
    end
end

end

macro layer_map(args...; kwargs...)
    Base.depwarn("`Lux.@layer_map` has been deprecated in favor of `Lux.Experimental.@layer_map`",
        Symbol("@layer_map"))
    return Experimental.@layer_map(args...; kwargs...)
end

for f in (:layer_map, :share_parameters, :FrozenLayer, :freeze, :unfreeze)
    msg = "`Lux.$(f)` has been deprecated in favor of  `Lux.Experimental.$(f)`"
    @eval begin
        $(f)(args...; kwargs...) = begin
            Base.depwarn($(msg), $(f))
            return Experimental.$(f)(args...; kwargs...)
        end
    end
end

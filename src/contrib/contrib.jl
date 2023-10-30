module Experimental

using ..Lux, LuxCore, LuxDeviceUtils, Random
import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer
import ..Lux: _merge,
    _pairs, initialstates, initialparameters, apply, NAME_TYPE, _getproperty
import ChainRulesCore as CRC
import ConcreteStructs: @concrete
import Functors: fmap
import TruncatedStacktraces

include("map.jl")
include("training.jl")
include("freeze.jl")
include("share_parameters.jl")
include("debug.jl")
include("stateful.jl")
include("compact.jl")

end

# Chain Rules for Certain Layers
## Only needed when the parameters are `nothing`
function CRC.rrule(::Type{<:Experimental.StatefulLuxLayer},
        model::AbstractExplicitLayer, ::Nothing, st)
    slayer = Experimental.StatefulLuxLayer(model, nothing, st)
    function ∇StatefulLuxLayer(Δ::Union{CRC.ZeroTangent, CRC.NoTangent})
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return slayer, ∇StatefulLuxLayer
end

# Deprecations for v0.6
module Training

using ..Experimental, Reexport
@reexport using ADTypes

for f in (:TrainState, :apply_gradients, :compute_gradients)
    msg = "`Lux.Training.$(f)` has been deprecated in favor of `Lux.Experimental.$(f)`"
    @eval begin
        function $(f)(args...; kwargs...)
            Base.depwarn($(msg), Symbol($(f)))
            return Experimental.$(f)(args...; kwargs...)
        end
    end
end

end

macro layer_map(f, l, ps, st)
    Base.depwarn("`Lux.@layer_map` has been deprecated in favor of `Lux.Experimental.@layer_map`",
        Symbol("@layer_map"))
    quote
        Experimental.layer_map($(esc(f)), $(esc(l)), $(esc(ps)), $(esc(st)), $(string(l)))
    end
end

for f in (:layer_map, :share_parameters, :FrozenLayer, :freeze, :unfreeze)
    msg = "`Lux.$(f)` has been deprecated in favor of `Lux.Experimental.$(f)`"
    @eval begin
        $(f)(args...; kwargs...) = begin
            Base.depwarn($(msg), Symbol($(f)))
            return Experimental.$(f)(args...; kwargs...)
        end
    end
end

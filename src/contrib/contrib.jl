module Experimental

import ..Lux
using ..Lux, LuxCore, LuxDeviceUtils, Random
using LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer
import ..Lux: _merge, _pairs, initialstates, initialparameters, apply

using ADTypes: ADTypes
using ChainRulesCore: ChainRulesCore
using ConcreteStructs: @concrete
using Functors: Functors, fmap, fmapstructure, functor
using Markdown: @doc_str
using Random: AbstractRNG, Random
using Setfield: Setfield

const CRC = ChainRulesCore

include("map.jl")
include("training.jl")
include("freeze.jl")
include("share_parameters.jl")
include("debug.jl")
include("deprecated.jl")

end

# Deprecations for v0.6
module Training

using ADTypes: AutoEnzyme, AutoReverseDiff, AutoTracker, AutoZygote
using ..Experimental: Experimental

for f in (:TrainState, :apply_gradients, :compute_gradients)
    msg = lazy"`Lux.Training.$(f)` has been deprecated in favor of `Lux.Experimental.$(f)`"
    @eval begin
        function $(f)(args...; kwargs...)
            Base.depwarn($(msg), Symbol($(f)))
            return Experimental.$(f)(args...; kwargs...)
        end
    end
end

export AutoEnzyme, AutoReverseDiff, AutoTracker, AutoZygote

end

macro layer_map(f, l, ps, st)
    Base.depwarn(
        "`Lux.@layer_map` has been deprecated in favor of `Lux.Experimental.@layer_map`",
        Symbol("@layer_map"))
    quote
        Experimental.layer_map($(esc(f)), $(esc(l)), $(esc(ps)), $(esc(st)), $(string(l)))
    end
end

for f in (:layer_map, :share_parameters, :FrozenLayer, :freeze, :unfreeze)
    msg = lazy"`Lux.$(f)` has been deprecated in favor of `Lux.Experimental.$(f)`"
    @eval begin
        $(f)(args...; kwargs...) = begin
            Base.depwarn($(msg), Symbol($(f)))
            return Experimental.$(f)(args...; kwargs...)
        end
    end
end

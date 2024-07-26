module Experimental

using ..Lux: Lux
using LuxCore: LuxCore, AbstractExplicitLayer, AbstractExplicitContainerLayer, apply

using ADTypes: ADTypes
using ArgCheck: @argcheck
using ChainRulesCore: ChainRulesCore
using Compat: @compat
using ConcreteStructs: @concrete
using FastClosures: @closure
using Functors: Functors, KeyPath, fmap_with_path, fmapstructure, functor
using Markdown: @doc_str
using Optimisers: Optimisers
using Random: AbstractRNG, Random
using Setfield: Setfield
using ..Training: Training

const CRC = ChainRulesCore

include("map.jl")
include("freeze.jl")
include("share_parameters.jl")
include("debug.jl")
include("deprecated.jl")

@compat public layer_map, @layer_map
@compat public FrozenLayer, freeze, unfreeze
@compat public share_parameters
@compat public DebugLayer, @debug_mode

end

# Deprecations for v1.0
macro layer_map(f, l, ps, st)
    Base.depwarn(
        "`Lux.@layer_map` has been deprecated in favor of `Lux.Experimental.@layer_map`",
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

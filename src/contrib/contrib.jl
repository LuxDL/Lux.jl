module Experimental

using ADTypes: ADTypes
using ChainRulesCore: ChainRulesCore
using SciMLPublic: @public
using ConcreteStructs: @concrete
using Functors: Functors, KeyPath, fmap_with_path, fmapstructure_with_path, functor
using Markdown: @doc_str
using Optimisers: Optimisers
using Random: AbstractRNG, Random
using Setfield: Setfield
using Static: StaticSymbol, StaticBool, True, known, static, dynamic

using ..Lux: Lux, Optional, CompactLuxLayer
using ..Utils: Utils, BoolType, SymbolType
using LuxCore:
    LuxCore, AbstractLuxLayer, AbstractLuxWrapperLayer, AbstractLuxContainerLayer, apply

const CRC = ChainRulesCore

include("map.jl")
include("freeze.jl")
include("share_parameters.jl")
include("debug.jl")

@public layer_map
@public FrozenLayer, freeze, unfreeze
@public share_parameters
@public DebugLayer
@public @debug_mode

end

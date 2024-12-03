module Experimental

using ADTypes: ADTypes
using ArgCheck: @argcheck
using ChainRulesCore: ChainRulesCore
using Compat: @compat
using ConcreteStructs: @concrete
using FastClosures: @closure
using Functors: Functors, KeyPath, fmap_with_path, fmapstructure_with_path, functor
using Markdown: @doc_str
using Optimisers: Optimisers
using Random: AbstractRNG, Random
using Setfield: Setfield
using Static: StaticSymbol, StaticBool, True, known, static, dynamic

using ..Lux: Lux, Optional
using ..Utils: Utils, BoolType, SymbolType
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxWrapperLayer,
               AbstractLuxContainerLayer, apply

const CRC = ChainRulesCore

include("map.jl")
include("freeze.jl")
include("share_parameters.jl")
include("debug.jl")

@compat public layer_map
@compat public FrozenLayer, freeze, unfreeze
@compat public share_parameters
@compat public DebugLayer, @debug_mode

end

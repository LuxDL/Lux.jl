module Losses # A huge chunk of this code has been derived from Flux.jl

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @thunk
    using FastClosures: @closure
end

const CRC = ChainRulesCore

include("utils.jl")
include("functions.jl")

end

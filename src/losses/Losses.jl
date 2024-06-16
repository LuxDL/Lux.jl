module Losses # A huge chunk of this code has been derived from Flux.jl

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @thunk
    using ConcreteStructs: @concrete
    using FastClosures: @closure
    using ..Lux: __unwrap_val
    using Markdown: @doc_str
    using LuxLib: logsoftmax, logsigmoid
    using Statistics: mean
end

const CRC = ChainRulesCore

abstract type AbstractLossFunction <: Function end

include("utils.jl")
include("loss_functions.jl")

export BinaryCrossEntropyLoss, BinaryFocalLoss, CrossEntropyLoss, FocalLoss, L1Loss, L2Loss,
       MAELoss, MSELoss, MSLELoss, SiameseContrastiveLoss, TverskyLoss

end

using .Losses
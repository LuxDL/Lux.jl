# Eventually the idea is to get `LossFunctions.jl` up to speed so that we don't need this
# sort of an implementation
module Losses # A huge chunk of this code has been derived from Flux.jl

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArrayInterface: fast_scalar_indexing
    using ArgCheck: @argcheck
    using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @thunk
    using Compat: @compat
    using ConcreteStructs: @concrete
    using FastClosures: @closure
    using ..Lux: __unwrap_val
    using LossFunctions: LossFunctions
    using LuxLib: logsoftmax, logsigmoid
    using Markdown: @doc_str
    using PartialFunctions: @$
    using Statistics: mean
end

const CRC = ChainRulesCore

abstract type AbstractLossFunction <: Function end

include("utils.jl")
include("loss_functions.jl")

@compat public xlogx, xlogy

export BinaryCrossEntropyLoss, BinaryFocalLoss, CrossEntropyLoss, DiceCoeffLoss, FocalLoss,
       HingeLoss, HuberLoss, KLDivergenceLoss, L1Loss, L2Loss, MAELoss, MSELoss, MSLELoss,
       PoissonLoss, SiameseContrastiveLoss, SquaredHingeLoss, TverskyLoss
export GenericLossFunction

end

using .Losses

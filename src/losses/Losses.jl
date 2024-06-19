# Eventually the idea is to create a package `DeepLearningLosses.jl` and move this
# functionality there and simply reexport it here.
module Losses # A huge chunk of this code has been derived from Flux.jl

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @thunk
    using Compat: @compat
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

@compat public xlogx, xlogy

export BinaryCrossEntropyLoss, BinaryFocalLoss, CrossEntropyLoss, DiceCoeffLoss, FocalLoss,
       HingeLoss, HuberLoss, KLDivergenceLoss, L1Loss, L2Loss, MAELoss, MSELoss, MSLELoss,
       PoissonLoss, SiameseContrastiveLoss, SquaredHingeLoss, TverskyLoss

end

using .Losses

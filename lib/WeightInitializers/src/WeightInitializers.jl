module WeightInitializers

using GPUArraysCore: @allowscalar
using LinearAlgebra: LinearAlgebra, Diagonal, qr, diagind
using Random: Random, AbstractRNG
using SpecialFunctions: SpecialFunctions, erfinv  # TODO: Move to Ext in v2.0
using Statistics: Statistics, std

include("partial.jl")
include("utils.jl")
include("initializers.jl")

export zeros64,
    ones64,
    rand64,
    randn64,
    zeros32,
    ones32,
    rand32,
    randn32,
    zeros16,
    ones16,
    rand16,
    randn16
export zerosC64,
    onesC64,
    randC64,
    randnC64,
    zerosC32,
    onesC32,
    randC32,
    randnC32,
    zerosC16,
    onesC16,
    randC16,
    randnC16
export glorot_normal, glorot_uniform
export kaiming_normal, kaiming_uniform
export truncated_normal, orthogonal, sparse_init, identity_init

end

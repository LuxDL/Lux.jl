module WeightInitializers

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ChainRulesCore, PartialFunctions, Random, SpecialFunctions, Statistics
end

include("utils.jl")
include("initializers.jl")

# Mark the functions as non-differentiable
for f in [
    :zeros64, :ones64, :rand64, :randn64, :zeros32, :ones32, :rand32, :randn32, :zeros16,
    :ones16, :rand16, :randn16, :zerosC64, :onesC64, :randC64, :randnC64, :zerosC32,
    :onesC32, :randC32, :randnC32, :zerosC16, :onesC16, :randC16, :randnC16, :glorot_normal,
    :glorot_uniform, :kaiming_normal, :kaiming_uniform, :truncated_normal]
    @eval @non_differentiable $(f)(::Any...)
end

export zeros64, ones64, rand64, randn64, zeros32, ones32, rand32, randn32, zeros16, ones16,
       rand16, randn16
export zerosC64, onesC64, randC64, randnC64, zerosC32, onesC32, randC32, randnC32, zerosC16,
       onesC16, randC16, randnC16
export glorot_normal, glorot_uniform
export kaiming_normal, kaiming_uniform
export truncated_normal

end

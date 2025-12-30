module ChainRulesCoreExt

using ChainRulesCore: @non_differentiable
using WeightInitializers: WeightInitializers, DeviceAgnostic

for f in [
    :zeros64,
    :ones64,
    :rand64,
    :randn64,
    :zeros32,
    :ones32,
    :rand32,
    :randn32,
    :zeros16,
    :ones16,
    :rand16,
    :randn16,
    :zerosC64,
    :onesC64,
    :randC64,
    :randnC64,
    :zerosC32,
    :onesC32,
    :randC32,
    :randnC32,
    :zerosC16,
    :onesC16,
    :randC16,
    :randnC16,
    :glorot_normal,
    :glorot_uniform,
    :kaiming_normal,
    :kaiming_uniform,
    :truncated_normal,
    :orthogonal,
    :sparse_init,
    :identity_init,
]
    @eval @non_differentiable WeightInitializers.$(f)(::Any...)
end

for f in (:zeros, :ones, :rand, :randn)
    @eval @non_differentiable DeviceAgnostic.$(f)(::Any...)
end

end

using Lux, Random, Test

include("../test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Chain(; d1=Dense(2 => 4, tanh), d2=Chain(; l1=Dense(4 => 2), l2=Dense(2 => 4)),
              d3=Dense(4 => 2))

ps, st = Lux.setup(rng, model)

sharing = (("d2.l2", "d1"), ("d3", "d2.l1"))

ps_1 = Lux.share_parameters(ps, sharing)

@test ps_1.d2.l2.weight === ps_1.d1.weight
@test ps_1.d2.l2.bias === ps_1.d1.bias
@test ps_1.d3.weight === ps_1.d2.l1.weight
@test ps_1.d3.bias === ps_1.d2.l1.bias

ps_new_1 = (; weight=randn(rng, Float32, 4, 2), bias=randn(rng, Float32, 4, 1))
ps_new_2 = (; weight=randn(rng, Float32, 2, 4), bias=randn(rng, Float32, 2, 1))

ps_2 = Lux.share_parameters(ps, sharing, (ps_new_1, ps_new_2))

@test ps_2.d2.l2.weight === ps_new_1.weight === ps_2.d1.weight
@test ps_2.d2.l2.bias === ps_new_1.bias === ps_2.d1.bias
@test ps_2.d3.weight === ps_new_2.weight === ps_2.d2.l1.weight
@test ps_2.d3.bias === ps_new_2.bias === ps_2.d2.l1.bias

# Mix in ComponentArray
ps_new_ca_1 = ComponentArray(ps_new_1)

ps_3 = Lux.share_parameters(ps, sharing, (ps_new_ca_1, ps_new_2))

@test ps_3.d2.l2.weight === ps_new_ca_1.weight === ps_3.d1.weight
@test ps_3.d2.l2.bias === ps_new_ca_1.bias === ps_3.d1.bias
@test ps_3.d3.weight === ps_new_2.weight === ps_3.d2.l1.weight
@test ps_3.d3.bias === ps_new_2.bias === ps_3.d2.l1.bias

# Input Checks
non_disjoint_sharing = (("d2.l2", "d1"), ("d1", "d2.l1"))
@test_throws ArgumentError Lux.share_parameters(ps, non_disjoint_sharing)
@test_throws ArgumentError Lux.share_parameters(ps, sharing, (ps_new_1,))

# Parameter Structure Mismatch
ps_new_1 = (; weight=randn(rng, Float32, 2, 4), bias=randn(rng, Float32, 4, 1))
ps_new_2 = (; weight=randn(rng, Float32, 2, 4), bias=randn(rng, Float32, 2, 1))

@test_throws ArgumentError Lux.share_parameters(ps, sharing, (ps_new_1, ps_new_2))

ps_new_ca_1 = ComponentArray(ps_new_1)

@test_throws ArgumentError Lux.share_parameters(ps, sharing, (ps_new_ca_1, ps_new_2))

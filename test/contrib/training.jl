import Lux, Optimisers, Random, Test

include("../test_utils.jl")

function _get_TrainState()
  rng = Random.MersenneTwister(0)

  model = Lux.Dense(3, 2)
  opt = Optimisers.Adam(0.01f0)

  tstate = Lux.Training.TrainState(Lux.replicate(rng), model, opt)

  x = randn(Lux.replicate(rng), Float32, (3, 1))

  return rng, tstate, model, opt, x
end

function _loss_function(model, ps, st, data)
  y, st = model(data, ps, st)
  return sum(y), st, ()
end

function test_TrainState_constructor()
  rng, tstate, model, opt, _ = _get_TrainState()

  ps, st = Lux.setup(Lux.replicate(rng), model)
  opt_st = Optimisers.setup(opt, tstate.parameters)

  Test.@test tstate.model == model
  Test.@test tstate.parameters == ps
  Test.@test tstate.states == st
  Test.@test isapprox(tstate.optimizer_state, opt_st)
  Test.@test tstate.step == 0

  return nothing
end

function test_abstract_vjp_interface()
  _, tstate, _, _, x = _get_TrainState()

  Test.@testset "NotImplemented" begin for vjp_rule in (Lux.Training.EnzymeVJP(),
                                                        Lux.Training.YotaVJP())
    Test.@test_throws ArgumentError Lux.Training.compute_gradients(vjp_rule, _loss_function,
                                                                   x, tstate)
  end end

  # Gradient Correctness should be tested in `test/autodiff.jl` and other parts of the
  # testing codebase. Here we only test that the API works.
  grads, _, _, _ = Test.@test_nowarn Lux.Training.compute_gradients(Lux.Training.ZygoteVJP(),
                                                                    _loss_function, x,
                                                                    tstate)
  tstate_ = Test.@test_nowarn Lux.Training.apply_gradients(tstate, grads)
  Test.@test tstate_.step == 1
  Test.@test tstate != tstate_

  return nothing
end

Test.@testset "TrainState" begin test_TrainState_constructor() end
Test.@testset "AbstractVJP" begin test_abstract_vjp_interface() end

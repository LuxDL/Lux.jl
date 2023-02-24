using Lux, Optimisers, Random, Test

include("../test_utils.jl")

function _get_TrainState()
    rng = MersenneTwister(0)

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

    @test tstate.model == model
    @test tstate.parameters == ps
    @test tstate.states == st
    @test isapprox(tstate.optimizer_state, opt_st)
    @test tstate.step == 0

    return nothing
end

function test_abstract_vjp_interface()
    _, tstate, _, _, x = _get_TrainState()

    @testset "NotImplemented" begin for vjp_rule in (Lux.Training.EnzymeVJP(),
                                                     Lux.Training.YotaVJP())
        @test_throws ArgumentError Lux.Training.compute_gradients(vjp_rule, _loss_function,
                                                                  x, tstate)
    end end

    # Gradient Correctness should be tested in `test/autodiff.jl` and other parts of the
    # testing codebase. Here we only test that the API works.
    for vjp_rule in (Lux.Training.ZygoteVJP(), Lux.Training.TrackerVJP())
        grads, _, _, _ = @test_nowarn Lux.Training.compute_gradients(vjp_rule,
                                                                     _loss_function, x,
                                                                     tstate)
        tstate_ = @test_nowarn Lux.Training.apply_gradients(tstate, grads)
        @test tstate_.step == 1
        @test tstate != tstate_
    end

    return nothing
end

@testset "TrainState" begin test_TrainState_constructor() end
@testset "AbstractVJP" begin test_abstract_vjp_interface() end

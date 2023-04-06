using Lux, Optimisers, Random, Test

include("../test_utils.jl")

function _loss_function(model, ps, st, data)
    y, st = model(data, ps, st)
    return sum(y), st, ()
end

@testset "$mode: TrainState" for (mode, aType, device, ongpu) in MODES
    rng = MersenneTwister(0)

    model = Dense(3, 2)
    opt = Adam(0.01f0)

    tstate = Lux.Training.TrainState(Lux.replicate(rng), model, opt;
                                     transform_variables=device)

    x = randn(Lux.replicate(rng), Float32, (3, 1)) |> aType

    ps, st = Lux.setup(Lux.replicate(rng), model) .|> device
    opt_st = Optimisers.setup(opt, tstate.parameters)

    @test check_approx(tstate.model, model)
    @test check_approx(tstate.parameters, ps)
    @test check_approx(tstate.states, st)
    @test check_approx(tstate.optimizer_state, opt_st)
    @test tstate.step == 0
end

@testset "$mode: AbstractVJP" for (mode, aType, device, ongpu) in MODES
    rng = MersenneTwister(0)

    model = Dense(3, 2)
    opt = Adam(0.01f0)

    tstate = Lux.Training.TrainState(Lux.replicate(rng), model, opt;
                                     transform_variables=device)

    x = randn(Lux.replicate(rng), Float32, (3, 1)) |> aType

    @testset "NotImplemented $(string(vjp_rule))" for vjp_rule in (Lux.Training.EnzymeVJP(),
                                                                   Lux.Training.YotaVJP())
        @test_throws ArgumentError Lux.Training.compute_gradients(vjp_rule, _loss_function,
                                                                  x, tstate)
    end

    for vjp_rule in (Lux.Training.ZygoteVJP(), Lux.Training.TrackerVJP())
        grads, _, _, _ = @test_nowarn Lux.Training.compute_gradients(vjp_rule,
                                                                     _loss_function, x,
                                                                     tstate)
        tstate_ = @test_nowarn Lux.Training.apply_gradients(tstate, grads)
        @test tstate_.step == 1
        @test tstate != tstate_
    end
end

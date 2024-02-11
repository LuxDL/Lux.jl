using ADTypes, Lux, Optimisers, Random, Test

include("../test_utils.jl")

function _loss_function(model, ps, st, data)
    y, st = model(data, ps, st)
    return sum(y), st, ()
end

@testset "$mode: TrainState" for (mode, aType, device, ongpu) in MODES
    rng = get_stable_rng(12345)

    model = Dense(3, 2)
    opt = Adam(0.01f0)

    tstate = Lux.Experimental.TrainState(Lux.replicate(rng), model, opt;
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

@testset "$mode: AbstractADTypes" for (mode, aType, device, ongpu) in MODES
    rng = get_stable_rng(12345)

    model = Dense(3, 2)
    opt = Adam(0.01f0)

    tstate = Lux.Experimental.TrainState(Lux.replicate(rng), model, opt;
        transform_variables=device)

    x = randn(Lux.replicate(rng), Float32, (3, 1)) |> aType

    @testset "NotImplemented $(string(ad))" for ad in (AutoEnzyme(), AutoReverseDiff())
        @test_throws ArgumentError Lux.Experimental.compute_gradients(
            ad, _loss_function, x,
            tstate)
    end

    for ad in (AutoZygote(), AutoTracker())
        grads, _, _, _ = @test_nowarn Lux.Experimental.compute_gradients(
            ad, _loss_function,
            x, tstate)
        tstate_ = @test_nowarn Lux.Experimental.apply_gradients(tstate, grads)
        @test tstate_.step == 1
        @test tstate != tstate_
    end
end

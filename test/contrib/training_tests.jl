@testitem "TrainState" setup=[SharedTestSetup] begin
    using Optimisers

    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        model = Dense(3, 2)
        opt = Adam(0.01f0)

        tstate = Lux.Experimental.TrainState(Lux.replicate(rng), model, opt)

        x = randn(Lux.replicate(rng), Float32, (3, 1))

        ps, st = Lux.setup(Lux.replicate(rng), model)
        opt_st = Optimisers.setup(opt, tstate.parameters)

        @test check_approx(tstate.model, model)
        @test check_approx(tstate.parameters, ps)
        @test check_approx(tstate.states, st)
        @test check_approx(tstate.optimizer_state, opt_st)
        @test tstate.step == 0
    end
end

@testitem "AbstractADTypes" setup=[SharedTestSetup] begin
    using ADTypes, Optimisers

    function _loss_function(model, ps, st, data)
        y, st = model(data, ps, st)
        return sum(y), st, ()
    end

    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        model = Dense(3, 2)
        opt = Adam(0.01f0)

        tstate = Lux.Experimental.TrainState(
            Lux.replicate(rng), model, opt; transform_variables=device)

        x = randn(Lux.replicate(rng), Float32, (3, 1)) |> aType

        for ad in (AutoZygote(), AutoTracker(), AutoReverseDiff())
            ongpu && ad isa AutoReverseDiff && continue

            grads, _, _, _ = Lux.Experimental.compute_gradients(
                ad, _loss_function, x, tstate)
            tstate_ = Lux.Experimental.apply_gradients(tstate, grads)
            @test tstate_.step == 1
            @test tstate != tstate_
        end
    end
end

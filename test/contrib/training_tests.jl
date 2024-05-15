@testitem "TrainState" setup=[SharedTestSetup] tags=[:contrib] begin
    using Optimisers

    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
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

@testitem "AbstractADTypes" setup=[SharedTestSetup] tags=[:contrib] begin
    using ADTypes, Optimisers, Enzyme

    function _loss_function(model, ps, st, data)
        y, st = model(data, ps, st)
        return sum(y), st, ()
    end

    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Dense(3, 2)
        opt = Adam(0.01f0)

        tstate = Lux.Experimental.TrainState(
            Lux.replicate(rng), model, opt; transform_variables=dev)

        x = randn(Lux.replicate(rng), Float32, (3, 1)) |> aType

        for ad in (AutoZygote(), AutoTracker(), AutoReverseDiff(), AutoEnzyme())
            ongpu && (ad isa AutoReverseDiff || ad isa AutoEnzyme) && continue

            grads, _, _, _ = Lux.Experimental.compute_gradients(
                ad, _loss_function, x, tstate)
            tstate_ = Lux.Experimental.apply_gradients(tstate, grads)
            @test tstate_.step == 1
            @test tstate != tstate_
        end
    end
end

@testitem "Training API" setup=[SharedTestSetup] tags=[:contrib] begin
    using ADTypes, Optimisers
    import Enzyme, Tracker, ReverseDiff, Zygote

    function mse(model, ps, st, data)
        x_data, y_data = data
        y, st_ = model(x_data, ps, st)
        return sum(abs2, y .- y_data), st_, ()
    end

    rng = get_stable_rng(12345)

    x_data = randn(rng, Float32, 4, 32)
    y_data = evalpoly.(x_data, ((1, 2, 3),)) .- evalpoly.(x_data, ((5, 2),))
    y_data = (y_data .- minimum(y_data)) ./ (maximum(y_data) - minimum(y_data))
    dataset = [(x_data[:, i], y_data[:, i]) for i in Iterators.partition(1:32, 8)]

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Chain(Dense(4, 32, tanh), BatchNorm(32),
            Dense(32, 32, tanh), BatchNorm(32), Dense(32, 4))
        dataset_ = [dev((x, y)) for (x, y) in dataset]
        opt = Adam(0.001f0)

        @testset "$(ad)" for ad in (
            AutoZygote(), AutoTracker(), AutoReverseDiff(), AutoEnzyme())
            ongpu && (ad isa AutoReverseDiff || ad isa AutoEnzyme) && continue

            tstate = Lux.Experimental.TrainState(
                Lux.replicate(rng), model, opt; transform_variables=dev)

            initial_loss = first(mse(model, tstate.parameters, tstate.states, dataset_[1]))

            for epoch in 1:100, (x, y) in dataset_
                grads, loss, _, tstate = Lux.Experimental.compute_gradients(
                    ad, mse, (x, y), tstate)
                tstate = Lux.Experimental.apply_gradients!(tstate, grads)
            end

            final_loss = first(mse(model, tstate.parameters, tstate.states, dataset_[1]))

            @test final_loss * 100 < initial_loss
        end
    end
end

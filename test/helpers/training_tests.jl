@testitem "TrainState" setup=[SharedTestSetup] tags=[:helpers] begin
    using Optimisers

    rng = StableRNG(12345)

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

@testitem "AbstractADTypes" setup=[SharedTestSetup] tags=[:helpers] begin
    using ADTypes, Optimisers, Enzyme

    function _loss_function(model, ps, st, data)
        y, st = model(data, ps, st)
        return sum(y), st, ()
    end

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Dense(3, 2)
        opt = Adam(0.01f0)

        tstate = Lux.Experimental.TrainState(
            Lux.replicate(rng), model, opt; transform_variables=dev)

        x = randn(Lux.replicate(rng), Float32, (3, 1)) |> aType

        for ad in (AutoZygote(), AutoTracker(), AutoReverseDiff(), AutoEnzyme())
            ongpu && (ad isa AutoReverseDiff || ad isa AutoEnzyme) && continue
            !LuxTestUtils.ENZYME_TESTING_ENABLED && ad isa AutoEnzyme && continue

            grads, _, _, _ = Lux.Experimental.compute_gradients(
                ad, _loss_function, x, tstate)
            tstate_ = Lux.Experimental.apply_gradients(tstate, grads)
            @test tstate_.step == 1
            @test tstate != tstate_
        end
    end
end

@testitem "Training API" setup=[SharedTestSetup] tags=[:helpers] begin
    using ADTypes, Optimisers
    import Enzyme, Tracker, ReverseDiff, Zygote

    mse = MSELoss()

    rng = StableRNG(12345)

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
            !LuxTestUtils.ENZYME_TESTING_ENABLED && ad isa AutoEnzyme && continue

            @test_throws ArgumentError Lux.Training.__maybe_implemented_compute_gradients(ad)

            @test_deprecated Lux.Experimental.TrainState(
                Lux.replicate(rng), model, opt; transform_variables=dev)

            tstate = Lux.Training.TrainState(
                Lux.replicate(rng), model, opt; transform_variables=dev)

            initial_loss = first(mse(model, tstate.parameters, tstate.states, dataset_[1]))

            for epoch in 1:100, (x, y) in dataset_
                grads, loss, _, tstate = Lux.Experimental.compute_gradients(
                    ad, mse, (x, y), tstate)
                tstate = Lux.Experimental.apply_gradients!(tstate, grads)
            end

            (x, y) = first(dataset_)
            @test_deprecated Lux.Experimental.compute_gradients(ad, mse, (x, y), tstate)
            grads, loss, _, tstate = Lux.Experimental.compute_gradients(
                ad, mse, (x, y), tstate)
            @test_deprecated Lux.Experimental.apply_gradients(tstate, grads)

            for epoch in 1:100, (x, y) in dataset_
                grads, loss, _, tstate = Lux.Experimental.single_train_step!(
                    ad, mse, (x, y), tstate)
            end

            for epoch in 1:100, (x, y) in dataset_
                grads, loss, _, tstate = Lux.Experimental.single_train_step(
                    ad, mse, (x, y), tstate)
            end

            # Test the adjust API
            tstate = Optimisers.adjust(tstate, 0.1f0)
            @test tstate.optimizer_state.layer_1.weight.rule.eta ≈ 0.1f0

            tstate = Optimisers.adjust(tstate; eta=0.5f0)
            @test tstate.optimizer_state.layer_1.weight.rule.eta ≈ 0.5f0

            Optimisers.adjust!(tstate, 0.01f0)
            @test tstate.optimizer_state.layer_1.weight.rule.eta ≈ 0.01f0

            Optimisers.adjust!(tstate; eta=0.11f0)
            @test tstate.optimizer_state.layer_1.weight.rule.eta ≈ 0.11f0

            final_loss = first(mse(model, tstate.parameters, tstate.states, dataset_[1]))

            @test final_loss * 100 < initial_loss
        end

        struct AutoCustomAD <: ADTypes.AbstractADType end

        tstate = Lux.Experimental.TrainState(
            Lux.replicate(rng), model, opt; transform_variables=dev)

        @test_throws ArgumentError Lux.Experimental.compute_gradients(
            AutoCustomAD(), mse, dataset_[1], tstate)
    end
end

@testitem "Enzyme: Invalidate Cache on State Update" setup=[SharedTestSetup] tags=[:helpers] begin
    using ADTypes, Optimisers
    using Enzyme

    if LuxTestUtils.ENZYME_TESTING_ENABLED
        Enzyme.API.runtimeActivity!(true)

        mse = MSELoss()
        function mse2(model, ps, st, (x, y))
            z, st = model(x, ps, st)
            return sum(abs2, z .- y), st, ()
        end

        rng = StableRNG(12345)

        model = Chain(Dense(4 => 3), VariationalHiddenDropout(0.5f0), Dense(3 => 4))
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 4, 32)
        opt = Adam(0.001f0)

        tstate = Lux.Experimental.TrainState(model, ps, st, opt)

        _, _, _, tstate_new = @inferred Lux.Experimental.compute_gradients(
            AutoEnzyme(), mse, (x, x), tstate)

        @test tstate_new.states !== tstate.states

        model = Chain(Dense(4 => 3), Dense(3 => 4))
        ps, st = Lux.setup(rng, model)

        tstate = Lux.Experimental.TrainState(model, ps, st, opt)

        _, _, _, tstate_new = @inferred Lux.Experimental.compute_gradients(
            AutoEnzyme(), mse, (x, x), tstate)

        @test @inferred(Lux.Experimental.compute_gradients(
            AutoEnzyme(), mse, (x, x), tstate_new)) isa Any

        _, _, _, tstate_new2 = @inferred Lux.Experimental.compute_gradients(
            AutoEnzyme(), mse2, (x, x), tstate_new)
        @test hasfield(typeof(tstate_new2.cache.extras), :forward)
        @test hasfield(typeof(tstate_new2.cache.extras), :reverse)
    else
        @test_broken false
    end
end

@testitem "Compiled ReverseDiff" setup=[SharedTestSetup] tags=[:helpers] begin
    using ADTypes, Optimisers, ReverseDiff

    mse1 = MSELoss()
    function mse2(model, ps, st, data)
        l, st_, stats = mse1(model, ps, st, data)
        return l, st_, (; data=2.0f0)
    end

    rng = StableRNG(12345)

    dataset = [(randn(rng, Float32, 4, 32), randn(rng, Float32, 4, 32)) for _ in 1:100]

    @testset "Unhandled Cases" begin
        model = Chain(Dense(4, 32, tanh), BatchNorm(32),
            Dense(32, 32, tanh), BatchNorm(32), Dense(32, 4))
        ps, st = Lux.setup(rng, model)

        tstate = Lux.Experimental.TrainState(model, ps, st, Adam(0.001f0))

        # Stateful models are not supported
        @test_throws ArgumentError Lux.Experimental.compute_gradients(
            AutoReverseDiff(; compile=true), mse1, dataset[1], tstate)

        model = Chain(Dense(4, 32, tanh), Dense(32, 32, tanh), Dense(32, 4))
        ps, st = Lux.setup(rng, model)

        tstate = Lux.Experimental.TrainState(model, ps, st, Adam(0.001f0))

        # Loss functions that return non-empty `stats` are not supported
        @test_throws ArgumentError Lux.Experimental.compute_gradients(
            AutoReverseDiff(; compile=true), mse2, dataset[1], tstate)

        struct StrangeModel <: Lux.AbstractExplicitLayer end

        function (m::StrangeModel)(x, ps, st)
            return x, (; new_state=0.0)
        end

        model = StrangeModel()
        ps, st = Lux.setup(rng, model)

        tstate = Lux.Experimental.TrainState(model, ps, st, Adam(0.001f0))

        # Stateful models are not supported
        @test_throws ArgumentError Lux.Experimental.compute_gradients(
            AutoReverseDiff(; compile=true), mse1, dataset[1], tstate)
    end

    model = Chain(Dense(4, 32, tanh), Dense(32, 32, tanh), Dense(32, 4))
    ps, st = Lux.setup(rng, model)

    tstate = Lux.Experimental.TrainState(model, ps, st, Adam(0.001f0))

    loss_initial = first(mse1(model, ps, st, dataset[1]))
    for i in 1:100
        for (x, y) in dataset
            _, _, _, tstate = Lux.Experimental.single_train_step!(
                AutoReverseDiff(; compile=true), mse1, (x, y), tstate)
        end
    end
    loss_final = first(mse1(model, tstate.parameters, tstate.states, dataset[1]))

    @test loss_final * 100 < loss_initial
end

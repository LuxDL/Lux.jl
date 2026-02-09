using ADTypes, Optimisers, Tracker, ReverseDiff, Mooncake, ComponentArrays, Enzyme

include("../../shared_testsetup.jl")

@testset "Training API" begin
    mse = MSELoss()

    rng = StableRNG(12345)

    x_data = randn(rng, Float32, 4, 32)
    y_data = evalpoly.(x_data, ((1, 2, 3),)) .- evalpoly.(x_data, ((5, 2),))
    y_data = (y_data .- minimum(y_data)) ./ (maximum(y_data) - minimum(y_data))
    dataset = [(x_data[:, i], y_data[:, i]) for i in Iterators.partition(1:32, 8)]

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Chain(
            Dense(4, 32, tanh),
            BatchNorm(32),
            Dense(32, 32, tanh),
            BatchNorm(32),
            Dense(32, 4),
        )
        dataset_ = [dev((x, y)) for (x, y) in dataset]
        opt = Adam(0.001f0)

        @testset "$(ad)" for ad in (
            AutoZygote(), AutoTracker(), AutoReverseDiff(), AutoEnzyme(), AutoMooncake()
        )
            ongpu &&
                (ad isa AutoReverseDiff || ad isa AutoEnzyme || ad isa AutoMooncake) &&
                continue
            !LuxTestUtils.ENZYME_TESTING_ENABLED[] && ad isa AutoEnzyme && continue
            !LuxTestUtils.ZYGOTE_TESTING_ENABLED[] && ad isa AutoZygote && continue

            function get_total_loss(model, tstate)
                loss = 0.0f0
                for (x, y) in dataset_
                    loss += mse(model, tstate.parameters, tstate.states, (x, y))[1]
                end
                return loss
            end

            @testset "compute_gradients + apply_gradients!" begin
                ps, st = dev(Lux.setup(rng, model))
                tstate = Training.TrainState(model, ps, st, opt)

                initial_loss = get_total_loss(model, tstate)

                for epoch in 1:1000, (x, y) in dataset_
                    grads, loss, _, tstate = allow_unstable() do
                        Training.compute_gradients(ad, mse, (x, y), tstate)
                    end
                    tstate = Training.apply_gradients!(tstate, grads)
                end

                final_loss = get_total_loss(model, tstate)
                @test final_loss * 100 < initial_loss
            end

            @testset "single_train_step!" begin
                ps, st = dev(Lux.setup(rng, model))
                tstate = Training.TrainState(model, ps, st, opt)

                initial_loss = get_total_loss(model, tstate)

                for epoch in 1:1000, (x, y) in dataset_
                    grads, loss, _, tstate = allow_unstable() do
                        Training.single_train_step!(ad, mse, (x, y), tstate)
                    end
                end

                final_loss = get_total_loss(model, tstate)
                @test final_loss * 100 < initial_loss
            end

            @testset "single_train_step" begin
                ps, st = dev(Lux.setup(rng, model))
                tstate = Training.TrainState(model, ps, st, opt)

                initial_loss = get_total_loss(model, tstate)

                for epoch in 1:1000, (x, y) in dataset_
                    grads, loss, _, tstate = allow_unstable() do
                        Training.single_train_step(ad, mse, (x, y), tstate)
                    end
                end

                final_loss = get_total_loss(model, tstate)
                @test final_loss * 100 < initial_loss
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
        end

        struct AutoCustomAD <: ADTypes.AbstractADType end

        ps, st = dev(Lux.setup(rng, model))
        tstate = Training.TrainState(model, ps, st, opt)

        @test_throws ArgumentError Training.compute_gradients(
            AutoCustomAD(), mse, dataset_[1], tstate
        )
    end
end

@testitem "TrainState" setup = [SharedTestSetup] tags = [:misc] begin
    using Optimisers

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Dense(3, 2)
        opt = Adam(0.01f0)
        ps, st = dev(Lux.setup(Lux.replicate(rng), model))

        tstate = Training.TrainState(model, ps, st, opt)

        x = randn(Lux.replicate(rng), Float32, (3, 1))
        opt_st = Optimisers.setup(opt, tstate.parameters)

        @test check_approx(tstate.model, model)
        @test check_approx(tstate.optimizer_state, opt_st)
        @test tstate.step == 0
    end
end

@testitem "AbstractADTypes" setup = [SharedTestSetup] tags = [:misc] begin
    using ADTypes, Optimisers, Tracker, ReverseDiff

    function _loss_function(model, ps, st, data)
        y, st = model(data, ps, st)
        return sum(y), st, ()
    end

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Dense(3, 2)
        opt = Adam(0.01f0)
        ps, st = dev(Lux.setup(rng, model))

        tstate = Training.TrainState(model, ps, st, opt)

        x = aType(randn(Lux.replicate(rng), Float32, (3, 1)))

        for ad in (AutoZygote(), AutoTracker(), AutoReverseDiff(), AutoEnzyme())
            ongpu && (ad isa AutoReverseDiff || ad isa AutoEnzyme) && continue
            !LuxTestUtils.ENZYME_TESTING_ENABLED && ad isa AutoEnzyme && continue

            grads, _, _, _ = Training.compute_gradients(ad, _loss_function, x, tstate)
            tstate_ = Training.apply_gradients(tstate, grads)
            @test tstate_.step == 1
            @test tstate != tstate_
        end
    end
end

@testitem "Training API" setup = [SharedTestSetup] tags = [:misc] begin
    using ADTypes, Optimisers
    using Mooncake, ReverseDiff, Tracker

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
            !LuxTestUtils.ENZYME_TESTING_ENABLED && ad isa AutoEnzyme && continue

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

@testitem "Training API ForwardDiff" setup = [SharedTestSetup] tags = [:misc] begin
    using ADTypes, Optimisers, ComponentArrays

    mse = MSELoss()

    rng = StableRNG(12345)

    x_data = randn(rng, Float32, 4, 32)
    y_data = evalpoly.(x_data, ((1, 2, 3),)) .- evalpoly.(x_data, ((5, 2),))
    y_data = (y_data .- minimum(y_data)) ./ (maximum(y_data) - minimum(y_data))
    dataset = [(x_data[:, i], y_data[:, i]) for i in Iterators.partition(1:32, 8)]

    model = Chain(
        Dense(4, 32, tanh), BatchNorm(32), Dense(32, 32, tanh), BatchNorm(32), Dense(32, 4)
    )

    dataset_ = [(x, y) for (x, y) in dataset]
    opt = Adam(0.001f0)

    ps, st = Lux.setup(rng, model)
    tstate = Training.TrainState(model, ComponentVector(ps), st, opt)

    initial_loss = first(
        mse(model, tstate.parameters, Lux.testmode(tstate.states), dataset_[1])
    )

    for epoch in 1:100, (x, y) in dataset_
        grads, loss, _, tstate = allow_unstable() do
            Training.compute_gradients(AutoForwardDiff(), mse, (x, y), tstate)
        end
        tstate = Training.apply_gradients!(tstate, grads)
    end

    for epoch in 1:100, (x, y) in dataset_
        grads, loss, _, tstate = allow_unstable() do
            Training.single_train_step!(AutoForwardDiff(), mse, (x, y), tstate)
        end
    end

    for epoch in 1:100, (x, y) in dataset_
        grads, loss, _, tstate = allow_unstable() do
            Training.single_train_step(AutoForwardDiff(), mse, (x, y), tstate)
        end
    end

    final_loss = first(mse(model, tstate.parameters, tstate.states, dataset_[1]))

    @test final_loss * 50 < initial_loss
end

@testitem "Training API Enzyme Runtime Mode" setup = [SharedTestSetup] tags = [:misc] skip =
    :(using LuxTestUtils; !LuxTestUtils.ENZYME_TESTING_ENABLED) begin
    using Lux, Random, Enzyme, Optimisers

    function makemodel(n)
        @compact(dense = Dense(n => 1; use_bias=true), b = ones(Float32, n)) do x
            @return dense(x .+ b)
        end
    end

    n_samples = 20
    x_dim = 10
    y_dim = 1

    model = makemodel(x_dim)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    W = randn(rng, Float32, y_dim, x_dim)
    b = randn(rng, Float32, y_dim)

    x_samples = randn(rng, Float32, x_dim, n_samples)
    y_samples = W * x_samples .+ b .+ 0.01f0 .* randn(rng, Float32, y_dim, n_samples)

    lossfn = MSELoss()

    function train_model!(model, ps, st, opt, nepochs::Int)
        tstate = Training.TrainState(model, ps, st, opt)
        for i in 1:nepochs
            grads, loss, _, tstate = Training.single_train_step!(
                AutoEnzyme(; mode=set_runtime_activity(Reverse)),
                lossfn,
                (x_samples, y_samples),
                tstate,
            )
        end
        return tstate.model, tstate.parameters, tstate.states
    end

    initial_loss = lossfn(first(model(x_samples, ps, st)), y_samples)

    model, ps, st = train_model!(model, ps, st, Descent(0.01f0), 10000)

    final_loss = lossfn(first(model(x_samples, ps, st)), y_samples)

    @test final_loss * 100 < initial_loss
end

@testitem "Enzyme: Invalidate Cache on State Update" setup = [SharedTestSetup] tags = [
    :misc
] skip = :(using LuxTestUtils; !LuxTestUtils.ENZYME_TESTING_ENABLED) begin
    using ADTypes, Optimisers

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

    tstate = Training.TrainState(model, ps, st, opt)

    _, _, _, tstate_new = Training.compute_gradients(AutoEnzyme(), mse, (x, x), tstate)

    @test tstate_new.states !== tstate.states

    model = Chain(Dense(4 => 3), Dense(3 => 4))
    ps, st = Lux.setup(rng, model)

    tstate = Training.TrainState(model, ps, st, opt)

    _, _, _, tstate_new = Training.compute_gradients(AutoEnzyme(), mse, (x, x), tstate)

    @test @inferred(Training.compute_gradients(AutoEnzyme(), mse, (x, x), tstate_new)) isa
        Any

    _, _, _, tstate_new2 = Training.compute_gradients(
        AutoEnzyme(), mse2, (x, x), tstate_new
    )
    @test hasfield(typeof(tstate_new2.cache.extras), :forward)
    @test hasfield(typeof(tstate_new2.cache.extras), :reverse)

    rng = StableRNG(12345)

    model = Chain(Dense(4 => 3), VariationalHiddenDropout(0.5f0), Dense(3 => 4))
    ps, st = Lux.setup(rng, model)
    x = randn(rng, Float32, 4, 32)
    opt = Adam(0.001f0)

    tstate = Training.TrainState(model, ps, st, opt)

    _, _, _, tstate_new = Training.compute_gradients(AutoEnzyme(), mse, (x, x), tstate)

    @test tstate_new.states !== tstate.states

    model = Chain(Dense(4 => 3), Dense(3 => 4))
    ps, st = Lux.setup(rng, model)

    tstate = Training.TrainState(model, ps, st, opt)

    _, _, _, tstate_new = Training.compute_gradients(AutoEnzyme(), mse, (x, x), tstate)

    @test @inferred(Training.compute_gradients(AutoEnzyme(), mse, (x, x), tstate_new)) isa
        Any

    _, _, _, tstate_new2 = Training.compute_gradients(
        AutoEnzyme(), mse2, (x, x), tstate_new
    )
    @test hasfield(typeof(tstate_new2.cache.extras), :forward)
    @test hasfield(typeof(tstate_new2.cache.extras), :reverse)
end

@testitem "Compiled ReverseDiff" setup = [SharedTestSetup] tags = [:misc] begin
    using ADTypes, Optimisers, ReverseDiff

    mse1 = MSELoss()
    function mse2(model, ps, st, data)
        l, st_, stats = mse1(model, ps, st, data)
        return l, st_, (; data=2.0f0)
    end

    rng = StableRNG(12345)

    dataset = [(randn(rng, Float32, 4, 32), randn(rng, Float32, 4, 32)) for _ in 1:100]

    @testset "Unhandled Cases" begin
        model = Chain(
            Dense(4, 32, tanh),
            BatchNorm(32),
            Dense(32, 32, tanh),
            BatchNorm(32),
            Dense(32, 4),
        )
        ps, st = Lux.setup(rng, model)

        tstate = Training.TrainState(model, ps, st, Adam(0.001f0))

        # Stateful models are not supported
        @test_throws ArgumentError Training.compute_gradients(
            AutoReverseDiff(; compile=true), mse1, dataset[1], tstate
        )

        model = Chain(Dense(4, 32, tanh), Dense(32, 32, tanh), Dense(32, 4))
        ps, st = Lux.setup(rng, model)

        tstate = Training.TrainState(model, ps, st, Adam(0.001f0))

        # Loss functions that return non-empty `stats` are not supported
        @test_throws ArgumentError Training.compute_gradients(
            AutoReverseDiff(; compile=true), mse2, dataset[1], tstate
        )

        struct StrangeModel <: Lux.AbstractLuxLayer end

        function (m::StrangeModel)(x, ps, st)
            return x, (; new_state=0.0)
        end

        model = StrangeModel()
        ps, st = Lux.setup(rng, model)

        tstate = Training.TrainState(model, ps, st, Adam(0.001f0))

        # Stateful models are not supported
        @test_throws ArgumentError Training.compute_gradients(
            AutoReverseDiff(; compile=true), mse1, dataset[1], tstate
        )
    end

    model = Chain(Dense(4, 32, tanh), Dense(32, 32, tanh), Dense(32, 4))
    ps, st = Lux.setup(rng, model)

    tstate = Training.TrainState(model, ps, st, Adam(0.001f0))

    loss_initial = first(mse1(model, ps, st, dataset[1]))
    for i in 1:100
        for (x, y) in dataset
            _, _, _, tstate = allow_unstable() do
                Training.single_train_step!(
                    AutoReverseDiff(; compile=true), mse1, (x, y), tstate
                )
            end
        end
    end
    loss_final = first(mse1(model, tstate.parameters, tstate.states, dataset[1]))

    @test loss_final * 100 < loss_initial
end

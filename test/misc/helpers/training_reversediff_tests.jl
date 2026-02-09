using ADTypes, Optimisers, ReverseDiff

include("../../shared_testsetup.jl")

@testset "Compiled ReverseDiff" begin
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

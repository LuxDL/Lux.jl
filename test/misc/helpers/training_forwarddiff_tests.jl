using ADTypes, Optimisers, ComponentArrays, ForwardDiff

include("../../shared_testsetup.jl")

@testset "Training API ForwardDiff" begin
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

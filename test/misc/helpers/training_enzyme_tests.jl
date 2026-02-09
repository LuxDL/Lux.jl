using ADTypes, Optimisers, Enzyme

include("../../shared_testsetup.jl")

@testset "Training API Enzyme Runtime Mode" begin
    if !LuxTestUtils.ENZYME_TESTING_ENABLED[]
        @test_broken false
        return nothing
    end

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

@testset "Enzyme: Invalidate Cache on State Update" begin
    if !LuxTestUtils.ENZYME_TESTING_ENABLED[]
        @test_broken false
        return nothing
    end

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

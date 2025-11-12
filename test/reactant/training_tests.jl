@testitem "Reactant: Training API" tags = [:reactant] setup = [SharedTestSetup] begin
    using Reactant, Optimisers

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        if mode == "amdgpu"
            @warn "Skipping AMDGPU tests for Reactant"
            continue
        end

        if ongpu
            Reactant.set_default_backend("gpu")
        else
            Reactant.set_default_backend("cpu")
        end

        xdev = reactant_device(; force=true)

        @testset "MLP Training: $(version)" for version in (:iip, :oop)
            model = Chain(
                Dense(2 => 32, gelu),
                BatchNorm(32),
                Dense(32 => 32, gelu),
                BatchNorm(32),
                Dense(32 => 2),
            )
            ps, st = xdev(Lux.setup(StableRNG(1234), model))

            x_ra = xdev(randn(Float32, 2, 32))
            y_ra = xdev(rand(Float32, 2, 32))

            inference_loss_fn =
                (xᵢ, yᵢ, mode, ps, st) -> begin
                    ŷᵢ, _ = model(xᵢ, ps, Lux.testmode(st))
                    return MSELoss()(ŷᵢ, yᵢ)
                end
            inference_loss_fn_compiled = @compile inference_loss_fn(
                x_ra, y_ra, model, ps, st
            )

            x = [rand(Float32, 2, 32) for _ in 1:32]
            y = [xᵢ .^ 2 for xᵢ in x]

            dataloader = DeviceIterator(xdev, zip(x, y))

            total_initial_loss = mapreduce(+, dataloader) do (xᵢ, yᵢ)
                inference_loss_fn_compiled(xᵢ, yᵢ, model, ps, st)
            end

            @testset for opt in (
                Descent(0.01f0),
                Momentum(0.01f0),
                Adam(0.01f0),
                AdamW(0.01f0),
                OptimiserChain(AccumGrad(5), Adam(0.01f0)),
            )
                ps, st = xdev(Lux.setup(StableRNG(1234), model))
                train_state = Training.TrainState(model, ps, st, opt)

                for epoch in 1:100, (xᵢ, yᵢ) in dataloader
                    grads, loss, stats, train_state = if version === :iip
                        Training.single_train_step!(
                            AutoEnzyme(), MSELoss(), (xᵢ, yᵢ), train_state
                        )
                    elseif version === :oop
                        Training.single_train_step(
                            AutoEnzyme(), MSELoss(), (xᵢ, yᵢ), train_state
                        )
                    else
                        error("Invalid version: $(version)")
                    end
                end

                total_final_loss = mapreduce(+, dataloader) do (xᵢ, yᵢ)
                    inference_loss_fn_compiled(
                        xᵢ, yᵢ, model, train_state.parameters, train_state.states
                    )
                end

                @test total_final_loss < 100 * total_initial_loss
            end
        end
    end
end

@testitem "Reactant Optimisers Patch: AccumGrad" tags = [:reactant] setup = [
    SharedTestSetup
] begin
    using Lux, Random, Reactant, Optimisers

    dev = reactant_device(; force=true)

    model = Chain(
        Dense(2 => 4, relu), Chain(Dense(4 => 2, relu; use_bias=false), Dense(2 => 1))
    )
    ps, st = Lux.setup(Random.default_rng(), model) |> dev

    x = randn(Float32, 2, 32) |> dev

    train_state = Training.TrainState(
        model, ps, st, OptimiserChain(AccumGrad(5), Descent(0.1))
    )
    st_opt = train_state.optimizer_state

    hlo = repr(@code_hlo(Optimisers.update(st_opt, ps, ps)))
    @test length(findall("stablehlo.if", hlo)) == (2 + 1 + 2) * 2
end

@testitem "Reactant Optimisers Patch: ClipNorm" tags = [:reactant] setup = [SharedTestSetup] begin
    using Lux, Random, Reactant, Optimisers

    dev = reactant_device(; force=true)

    model = Chain(
        Dense(2 => 4, relu), Chain(Dense(4 => 2, relu; use_bias=false), Dense(2 => 2))
    )
    ps, st = Lux.setup(Random.default_rng(), model) |> dev

    x = randn(Float32, 2, 32) |> dev

    train_state = Training.TrainState(
        model, ps, st, OptimiserChain(ClipNorm(0.5), Descent(0.1))
    )

    _, loss, stats, ts = Training.single_train_step(
        AutoEnzyme(), MSELoss(), (x, x), train_state; return_gradients=Val(false)
    )
    @test loss isa Number
end

@testitem "Reactant Distributed: Training API" tags = [:reactant] setup = [SharedTestSetup] begin
    using Lux, Random, Reactant, Optimisers

    ndevices = length(Reactant.devices())

    if ndevices ≥ 8 && Reactant.XLA.runtime() isa Val{:IFRT}
        mesh = Sharding.Mesh(reshape(Reactant.devices()[1:8], (2, 4)), (:model, :batch))

        model_device = reactant_device(;
            sharding=Sharding.DimsSharding(mesh, (-2,), (:model,))
        )
        batch_device = reactant_device(;
            sharding=Sharding.DimsSharding(mesh, (-1,), (:batch,))
        )

        model = Chain(
            Chain(Dense(4 => 32), BatchNorm(32, relu)),
            Chain(Dense(32 => 32), BatchNorm(32, relu)),
            Dense(32 => 4),
        )
        ps, st = Lux.setup(Random.default_rng(), model) |> model_device

        x = rand(Float32, 4, 128) |> batch_device
        y = rand(Float32, 4, 128) |> batch_device

        train_state = Training.TrainState(model, ps, st, Adam(0.001f0))

        _, loss, _, train_state = Training.single_train_step(
            AutoEnzyme(), MSELoss(), (x, y), train_state
        )
        @test loss isa Reactant.ConcreteRNumber
        @test length(Reactant.XLA.devices(Reactant.XLA.sharding(loss.data))) == 8

        _, loss, _, train_state = Training.single_train_step(
            AutoEnzyme(), MSELoss(), (x, y), train_state
        )
        @test loss isa Reactant.ConcreteRNumber
        @test length(Reactant.XLA.devices(Reactant.XLA.sharding(loss.data))) == 8
    end
end

@testitem "Reactant.Compiler.Thunk in TrainState" tags = [:reactant] setup = [
    SharedTestSetup
] begin
    using Lux, Random, Reactant, Optimisers

    rdev = reactant_device(; force=true)

    model = Dense(10, 10)
    ps, st = Lux.setup(Random.default_rng(), model) |> rdev
    x = rand(10) |> rdev

    model_compiled = @compile model(x, ps, st)

    @test_throws ArgumentError Training.TrainState(model_compiled, ps, st, Adam())
end

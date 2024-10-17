@testitem "Reactant: Training API" tags=[:reactant] setup=[SharedTestSetup] begin
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

        xdev = xla_device(; force=true)

        @testset "MLP Training: $(version)" for version in (:iip, :oop)
            model = Chain(
                Dense(2 => 32, gelu),
                Dense(32 => 32, gelu),
                Dense(32 => 2)
            )
            ps, st = Lux.setup(StableRNG(1234), model) |> xdev

            x_ra = randn(Float32, 2, 32) |> xdev

            inference_fn = @compile model(x_ra, ps, Lux.testmode(st))

            x = [rand(Float32, 2, 32) for _ in 1:32]
            y = [xᵢ .^ 2 for xᵢ in x]

            dataloader = DeviceIterator(xdev, zip(x, y))

            total_initial_loss = mapreduce(+, dataloader) do (xᵢ, yᵢ)
                ŷᵢ, _ = inference_fn(xᵢ, ps, Lux.testmode(st))
                return MSELoss()(ŷᵢ, yᵢ)
            end

            train_state = Training.TrainState(model, ps, st, Adam(0.01f0))

            for epoch in 1:100, (xᵢ, yᵢ) in dataloader

                grads, loss,
                stats,
                train_state = if version === :iip
                    Training.single_train_step!(
                        AutoEnzyme(), MSELoss(), (xᵢ, yᵢ), train_state)
                elseif version === :oop
                    Training.single_train_step(
                        AutoEnzyme(), MSELoss(), (xᵢ, yᵢ), train_state)
                else
                    error("Invalid version: $(version)")
                end
            end

            total_final_loss = mapreduce(+, dataloader) do (xᵢ, yᵢ)
                ŷᵢ, _ = inference_fn(xᵢ, train_state.parameters, Lux.testmode(st))
                return MSELoss()(ŷᵢ, yᵢ)
            end

            @test total_final_loss < 100 * total_initial_loss
        end
    end
end

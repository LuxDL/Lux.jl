@testitem "Batched Jacobian" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ComponentArrays, ForwardDiff, Zygote

    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # FIXME: AMDGPU takes too long right now
        mode === "AMDGPU" && continue

        models = (
            Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
                Conv((3, 3), 4 => 2, gelu; pad=SamePad()), FlattenLayer(), Dense(18 => 2)),
            Chain(Dense(2, 4, gelu), Dense(4, 2)))
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (model, X) in zip(models, Xs)
            ps, st = Lux.setup(rng, model) |> dev
            smodel = StatefulLuxLayer(model, ps, st)

            J1 = ForwardDiff.jacobian(smodel, X)

            J2 = batched_jacobian(smodel, AutoForwardDiff(), X)
            J2_mat = mapreduce(Base.Fix1(Lux.__maybe_batched_row, J2),
                hcat, 1:(size(J2, 1) * size(J2, 3)))'

            @test J1≈J2_mat atol=1.0e-5 rtol=1.0e-5

            ps = ps |> cpu_device() |> ComponentArray |> dev

            smodel = StatefulLuxLayer(model, ps, st)

            J3 = batched_jacobian(smodel, AutoForwardDiff(), X)

            @test J2≈J3 atol=1.0e-5 rtol=1.0e-5
        end

        @testset "Issue #636 Chunksize Specialization" begin
            for N in (2, 4, 8, 11, 12, 50, 51)
                model = @compact(; potential=Dense(N => N, gelu)) do x
                    return batched_jacobian(potential, AutoForwardDiff(), x)
                end

                ps, st = Lux.setup(Random.default_rng(), model) |> dev

                x = randn(Float32, N, 3) |> dev
                @test_nowarn model(x, ps, st)
            end
        end
    end
end

@testitem "Nested AD: Batched Jacobian" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ComponentArrays, ForwardDiff, Zygote

    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # FIXME: AMDGPU takes too long right now
        mode === "AMDGPU" && continue

        models = (
            Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
                Conv((3, 3), 4 => 2, gelu; pad=SamePad()), FlattenLayer(), Dense(18 => 2)),
            Chain(Dense(2, 4, gelu), Dense(4, 2)))
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (model, X) in zip(models, Xs)
            ps, st = Lux.setup(rng, model) |> dev

            function loss_function_batched(model, x, ps, st)
                smodel = StatefulLuxLayer(model, ps, st)
                J = batched_jacobian(smodel, AutoForwardDiff(), x)
                return sum(abs2, J)
            end

            function loss_function_simple(model, x, ps, st)
                smodel = StatefulLuxLayer(model, ps, st)
                J = ForwardDiff.jacobian(smodel, x)
                return sum(abs2, J)
            end

            @test_nowarn loss_function_batched(model, X, ps, st)
            @test loss_function_batched(model, X, ps, st) ≈
                  loss_function_simple(model, X, ps, st)

            _, ∂x_batched, ∂ps_batched, _ = Zygote.gradient(
                loss_function_batched, model, X, ps, st)
            _, ∂x_simple, ∂ps_simple, _ = Zygote.gradient(
                loss_function_simple, model, X, ps, st)

            @test ∂x_batched≈∂x_simple atol=1.0e-5 rtol=1.0e-5
            @test check_approx(∂ps_batched, ∂ps_simple; atol=1.0e-5, rtol=1.0e-5)

            ps = ps |> cpu_device() |> ComponentArray |> dev

            _, ∂x_batched2, ∂ps_batched2, _ = Zygote.gradient(
                loss_function_batched, model, X, ps, st)

            @test ∂x_batched2≈∂x_batched atol=1.0e-5 rtol=1.0e-5
            @test check_approx(∂ps_batched2, ∂ps_batched; atol=1.0e-5, rtol=1.0e-5)
        end
    end
end

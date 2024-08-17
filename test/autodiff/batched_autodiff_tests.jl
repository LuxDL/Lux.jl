@testitem "Batched Jacobian" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ComponentArrays, ForwardDiff, Zygote

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        models = (
            Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
                Conv((3, 3), 4 => 2, gelu; pad=SamePad()), FlattenLayer(), Dense(18 => 2)),
            Chain(Dense(2, 4, gelu), Dense(4, 2)))
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (model, X) in zip(models, Xs)
            ps, st = Lux.setup(rng, model) |> dev
            smodel = StatefulLuxLayer{true}(model, ps, st)

            J1 = ForwardDiff.jacobian(smodel, X)

            @testset "$(backend)" for backend in (AutoZygote(), AutoForwardDiff())
                J2 = batched_jacobian(smodel, backend, X)
                J2_mat = mapreduce(Base.Fix1(Lux.AutoDiffInternalImpl.batched_row, J2),
                    hcat, 1:(size(J2, 1) * size(J2, 3)))'

                @test J1≈J2_mat atol=1.0e-3 rtol=1.0e-3

                ps = ps |> cpu_device() |> ComponentArray |> dev

                smodel = StatefulLuxLayer{true}(model, ps, st)

                J3 = batched_jacobian(smodel, backend, X)

                @test J2≈J3 atol=1.0e-3 rtol=1.0e-3
            end
        end

        @testset "Issue #636 Chunksize Specialization" begin
            for N in (2, 4, 8, 11, 12, 50, 51), backend in (AutoZygote(), AutoForwardDiff())
                model = @compact(; potential=Dense(N => N, gelu), backend=backend) do x
                    @return batched_jacobian(potential, backend, x)
                end

                ps, st = Lux.setup(Random.default_rng(), model) |> dev

                x = randn(Float32, N, 3) |> dev
                @test first(model(x, ps, st)) isa AbstractArray{<:Any, 3}
            end
        end

        @testset "Simple Batched Jacobian" begin
            # Without any Lux bs just plain old batched jacobian
            ftest(x) = x .^ 2
            x = reshape(Float32.(1:6), 2, 3) |> dev

            Jx_true = zeros(Float32, 2, 2, 3)
            Jx_true[1, 1, 1] = 2
            Jx_true[2, 2, 1] = 4
            Jx_true[1, 1, 2] = 6
            Jx_true[2, 2, 2] = 8
            Jx_true[1, 1, 3] = 10
            Jx_true[2, 2, 3] = 12
            Jx_true = Jx_true |> dev

            Jx_fdiff = batched_jacobian(ftest, AutoForwardDiff(), x)
            @test Jx_fdiff ≈ Jx_true

            Jx_zygote = batched_jacobian(ftest, AutoZygote(), x)
            @test Jx_zygote ≈ Jx_true

            fincorrect(x) = x[:, 1]
            x = reshape(Float32.(1:6), 2, 3) |> dev

            @test_throws ArgumentError batched_jacobian(fincorrect, AutoForwardDiff(), x)
            @test_throws AssertionError batched_jacobian(fincorrect, AutoZygote(), x)
        end
    end
end

@testitem "Nested AD: Batched Jacobian" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ComponentArrays, ForwardDiff, Zygote

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        models = (
            Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
                Conv((3, 3), 4 => 2, gelu; pad=SamePad()), FlattenLayer(), Dense(18 => 2)),
            Chain(Dense(2, 4, gelu), Dense(4, 2)))
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (model, X) in zip(models, Xs), backend in (AutoZygote(), AutoForwardDiff())
            model = maybe_rewrite_to_crosscor(mode, model)
            ps, st = Lux.setup(rng, model) |> dev

            function loss_function_batched(model, x, ps, st)
                smodel = StatefulLuxLayer{true}(model, ps, st)
                J = batched_jacobian(smodel, backend, x)
                return sum(abs2, J)
            end

            function loss_function_simple(model, x, ps, st)
                smodel = StatefulLuxLayer{true}(model, ps, st)
                J = ForwardDiff.jacobian(smodel, x)
                return sum(abs2, J)
            end

            @test loss_function_batched(model, X, ps, st) isa Number
            @test loss_function_batched(model, X, ps, st) ≈
                  loss_function_simple(model, X, ps, st)

            _, ∂x_batched, ∂ps_batched, _ = Zygote.gradient(
                loss_function_batched, model, X, ps, st)
            _, ∂x_simple, ∂ps_simple, _ = Zygote.gradient(
                loss_function_simple, model, X, ps, st)

            @test ∂x_batched≈∂x_simple atol=1.0e-3 rtol=1.0e-3
            @test check_approx(∂ps_batched, ∂ps_simple; atol=1.0e-3, rtol=1.0e-3)

            ps = ps |> cpu_device() |> ComponentArray |> dev

            _, ∂x_batched2, ∂ps_batched2, _ = Zygote.gradient(
                loss_function_batched, model, X, ps, st)

            @test ∂x_batched2≈∂x_batched atol=1.0e-3 rtol=1.0e-3
            @test check_approx(∂ps_batched2, ∂ps_batched; atol=1.0e-3, rtol=1.0e-3)
        end
    end
end

@testitem "Nested AD: Batched Jacobian Single Input" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ForwardDiff, Zygote

    rng = StableRNG(12345)
    sq_fn(x) = x .^ 2

    sumabs2_fd(x) = sum(abs2, batched_jacobian(sq_fn, AutoForwardDiff(), x))
    sumabs2_zyg(x) = sum(abs2, batched_jacobian(sq_fn, AutoZygote(), x))

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = rand(rng, Float32, 4, 2) |> aType

        @test sumabs2_fd(x) ≈ sumabs2_zyg(x)

        ∂x1 = Zygote.gradient(sumabs2_zyg, x)[1]
        ∂x2 = Zygote.gradient(sumabs2_fd, x)[1]
        ∂x_gt = ForwardDiff.gradient(sumabs2_fd, x)

        @test ∂x1≈∂x_gt atol=1.0e-3 rtol=1.0e-3
        @test ∂x2≈∂x_gt atol=1.0e-3 rtol=1.0e-3
    end
end

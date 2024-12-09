@testitem "Batched Jacobian" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ComponentArrays, ForwardDiff, Zygote, ADTypes

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        models = (
            Chain(
                Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
                Conv((3, 3), 4 => 2, gelu; pad=SamePad()),
                FlattenLayer(), Dense(18 => 2)
            ),
            Chain(Dense(2, 4, gelu), Dense(4, 2))
        )
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (i, (model, X)) in enumerate(zip(models, Xs))
            ps, st = Lux.setup(rng, model) |> dev
            smodel = StatefulLuxLayer{true}(model, ps, st)

            J1 = allow_unstable() do
                ForwardDiff.jacobian(smodel, X)
            end

            @testset for backend in (
                AutoZygote(), AutoForwardDiff(),
                AutoEnzyme(;
                    mode=Enzyme.Forward, function_annotation=Enzyme.Const
                ),
                AutoEnzyme(;
                    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
                    function_annotation=Enzyme.Const
                )
            )
                # Forward rules for Enzyme is currently not implemented for several Ops
                i == 1 && backend isa AutoEnzyme &&
                    ADTypes.mode(backend) isa ADTypes.ForwardMode && continue

                J2 = allow_unstable() do
                    batched_jacobian(smodel, backend, X)
                end
                J2_mat = mapreduce(Base.Fix1(Lux.AutoDiffInternalImpl.batched_row, J2),
                    hcat, 1:(size(J2, 1) * size(J2, 3)))'

                @test J1≈J2_mat atol=1.0e-3 rtol=1.0e-3

                ps = ps |> cpu_device() |> ComponentArray |> dev

                smodel = StatefulLuxLayer{true}(model, ps, st)

                J3 = allow_unstable() do
                    batched_jacobian(smodel, backend, X)
                end

                @test J2≈J3 atol=1.0e-3 rtol=1.0e-3
            end
        end

        @testset "Issue #636 Chunksize Specialization" begin
            for N in (2, 4, 8, 11, 12, 50, 51),
                backend in (
                    AutoZygote(), AutoForwardDiff(), AutoEnzyme(),
                    AutoEnzyme(; mode=Enzyme.Reverse)
                )

                ongpu && backend isa AutoEnzyme && continue

                model = @compact(; potential=Dense(N => N, gelu), backend=backend) do x
                    @return allow_unstable() do
                        batched_jacobian(potential, backend, x)
                    end
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

            Jx_fdiff = allow_unstable() do
                batched_jacobian(ftest, AutoForwardDiff(), x)
            end
            @test Jx_fdiff ≈ Jx_true

            Jx_zygote = allow_unstable() do
                batched_jacobian(ftest, AutoZygote(), x)
            end
            @test Jx_zygote ≈ Jx_true

            if !ongpu
                Jx_enzyme = allow_unstable() do
                    batched_jacobian(ftest, AutoEnzyme(), x)
                end
                @test Jx_enzyme ≈ Jx_true
            end

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

            @test allow_unstable() do
                loss_function_batched(model, X, ps, st)
            end isa Number
            @test allow_unstable() do
                loss_function_batched(model, X, ps, st)
            end ≈ allow_unstable() do
                loss_function_simple(model, X, ps, st)
            end

            _, ∂x_batched, ∂ps_batched, _ = allow_unstable() do
                Zygote.gradient(loss_function_batched, model, X, ps, st)
            end
            _, ∂x_simple, ∂ps_simple, _ = allow_unstable() do
                Zygote.gradient(loss_function_simple, model, X, ps, st)
            end

            @test ∂x_batched≈∂x_simple atol=1.0e-3 rtol=1.0e-3
            @test check_approx(∂ps_batched, ∂ps_simple; atol=1.0e-3, rtol=1.0e-3)

            ps = ps |> cpu_device() |> ComponentArray |> dev

            _, ∂x_batched2, ∂ps_batched2, _ = allow_unstable() do
                Zygote.gradient(loss_function_batched, model, X, ps, st)
            end

            @test ∂x_batched2≈∂x_batched atol=1.0e-3 rtol=1.0e-3
            @test check_approx(∂ps_batched2, ∂ps_batched; atol=1.0e-3, rtol=1.0e-3)
        end
    end
end

@testitem "Nested AD: Batched Jacobian Single Input" setup=[SharedTestSetup] tags=[:autodiff] begin
    using Zygote, Tracker, ReverseDiff

    rng = StableRNG(12345)
    sq_fn(x) = x .^ 2

    sumabs2_fd(x) = sum(abs2, batched_jacobian(sq_fn, AutoForwardDiff(), x))
    sumabs2_zyg(x) = sum(abs2, batched_jacobian(sq_fn, AutoZygote(), x))

    true_gradient(x) = 8 .* x

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = rand(rng, Float32, 4, 2) |> aType

        @test sumabs2_fd(x) ≈ sumabs2_zyg(x)

        ∂x1_zyg = allow_unstable() do
            only(Zygote.gradient(sumabs2_zyg, x))
        end
        ∂x1_tr = allow_unstable() do
            only(Tracker.gradient(sumabs2_zyg, x))
        end
        ∂x2_zyg = allow_unstable() do
            only(Zygote.gradient(sumabs2_fd, x))
        end
        ∂x2_tr = allow_unstable() do
            only(Tracker.gradient(sumabs2_fd, x))
        end

        ∂x_gt = true_gradient(x)

        @test ∂x1_zyg≈∂x_gt atol=1.0e-3 rtol=1.0e-3
        @test ∂x1_tr≈∂x_gt atol=1.0e-3 rtol=1.0e-3
        @test ∂x2_zyg≈∂x_gt atol=1.0e-3 rtol=1.0e-3
        @test ∂x2_tr≈∂x_gt atol=1.0e-3 rtol=1.0e-3

        ongpu && continue

        ∂x1_rdiff = allow_unstable() do
            ReverseDiff.gradient(sumabs2_zyg, x)
        end
        ∂x2_rdiff = allow_unstable() do
            ReverseDiff.gradient(sumabs2_fd, x)
        end

        @test ∂x1_rdiff≈∂x_gt atol=1.0e-3 rtol=1.0e-3
        @test ∂x2_rdiff≈∂x_gt atol=1.0e-3 rtol=1.0e-3
    end
end

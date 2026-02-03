using ComponentArrays, ForwardDiff, Zygote, Tracker, ReverseDiff

include("../shared_testsetup.jl")

@testset "Nested AD: Batched Jacobian" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        models = (
            Chain(
                Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
                Conv((3, 3), 4 => 2, gelu; pad=SamePad()),
                FlattenLayer(),
                Dense(18 => 2),
            ),
            Chain(Dense(2, 4, gelu), Dense(4, 2)),
        )
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (model, X) in zip(models, Xs), backend in (AutoZygote(), AutoForwardDiff())
            model = maybe_rewrite_to_crosscor(mode, model)
            ps, st = dev(Lux.setup(rng, model))

            function loss_function_batched(model, x, ps, st)
                smodel = StatefulLuxLayer(model, ps, st)
                J = batched_jacobian(smodel, backend, x)
                return sum(abs2, J)
            end

            function loss_function_simple(model, x, ps, st)
                smodel = StatefulLuxLayer(model, ps, st)
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

            @test ∂x_batched ≈ ∂x_simple atol = 1.0e-3 rtol = 1.0e-3
            @test check_approx(∂ps_batched, ∂ps_simple; atol=1.0e-3, rtol=1.0e-3)

            ps = dev(ComponentArray(cpu_device()(ps)))

            _, ∂x_batched2, ∂ps_batched2, _ = allow_unstable() do
                Zygote.gradient(loss_function_batched, model, X, ps, st)
            end

            @test ∂x_batched2 ≈ ∂x_batched atol = 1.0e-3 rtol = 1.0e-3
            @test check_approx(∂ps_batched2, ∂ps_batched; atol=1.0e-3, rtol=1.0e-3)
        end
    end
end

@testset "Nested AD: Batched Jacobian Single Input" begin
    rng = StableRNG(12345)
    sq_fn(x) = x .^ 2

    sumabs2_fd(x) = sum(abs2, batched_jacobian(sq_fn, AutoForwardDiff(), x))
    sumabs2_zyg(x) = sum(abs2, batched_jacobian(sq_fn, AutoZygote(), x))

    true_gradient(x) = 8 .* x

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = aType(rand(rng, Float32, 4, 2))

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

        @test ∂x1_zyg ≈ ∂x_gt atol = 1.0e-3 rtol = 1.0e-3
        @test ∂x1_tr ≈ ∂x_gt atol = 1.0e-3 rtol = 1.0e-3
        @test ∂x2_zyg ≈ ∂x_gt atol = 1.0e-3 rtol = 1.0e-3
        @test ∂x2_tr ≈ ∂x_gt atol = 1.0e-3 rtol = 1.0e-3

        ongpu && continue

        ∂x1_rdiff = allow_unstable() do
            ReverseDiff.gradient(sumabs2_zyg, x)
        end
        ∂x2_rdiff = allow_unstable() do
            ReverseDiff.gradient(sumabs2_fd, x)
        end

        @test ∂x1_rdiff ≈ ∂x_gt atol = 1.0e-3 rtol = 1.0e-3
        @test ∂x2_rdiff ≈ ∂x_gt atol = 1.0e-3 rtol = 1.0e-3
    end
end

@testitem "Nested ForwardDiff over Zygote" setup=[SharedTestSetup] tags=[:others] begin
    using ComponentArrays, FiniteDifferences, ForwardDiff, LinearAlgebra, Zygote

    Base.isfinite(::Nothing) = true

    rng = get_stable_rng()

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # FIXME: AMDGPU takes too long right now
        mode === "AMDGPU" && continue

        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)),
            aType(randn(rng, Float32, 2, 4)), aType(randn(rng, Float32, 3, 3, 2, 4)))
        models = (
            Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()), BatchNorm(4),
                Conv((3, 3), 4 => 2, gelu; pad=SamePad()),
                BatchNorm(2), FlattenLayer(), Dense(18 => 2)),
            Chain(Dense(2, 4, gelu), Dense(4, 2)),
            Chain(Dense(2, 4, gelu), BatchNorm(4, sigmoid), Dense(4, 2)),
            Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()), BatchNorm(4),
                Conv((3, 3), 4 => 2, tanh; pad=SamePad()),
                BatchNorm(2), FlattenLayer(), Dense(18 => 1)))

        for (X, model) in zip(Xs, models)
            ps, st = Lux.setup(rng, model) |> dev
            X = X |> aType

            # smodel | ForwardDiff.jacobian
            loss_function1 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer(model, ps, st)
                return sum(abs2, ForwardDiff.jacobian(smodel, x))
            end

            # smodel | Zygote.jacobian
            loss_function2 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer(model, ps, st)
                return sum(abs2, only(Zygote.jacobian(smodel, x)))
            end

            # sum(abs2) ∘ smodel | ForwardDiff.gradient
            loss_function3 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer(model, ps, st)
                return sum(abs2, ForwardDiff.gradient(Base.Fix1(sum, abs2) ∘ smodel, x))
            end

            # sum(abs2) ∘ smodel | Zygote.gradient
            loss_function4 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer(model, ps, st)
                return sum(abs2, only(Zygote.gradient(Base.Fix1(sum, abs2) ∘ smodel, x)))
            end

            loss_fns = ongpu ? (loss_function2, loss_function4) :
                       (loss_function1, loss_function2, loss_function3, loss_function4)

            for loss_fn in loss_fns
                @test_nowarn loss_fn(model, X, ps, st)
                @test loss_fn(model, X, ps, st) isa Number

                _, ∂x, ∂ps, _ = Zygote.gradient(loss_fn, model, X, ps, st)

                @test ∂x !== nothing && !iszero(∂x) && all(isfinite, ∂x)
                @test ∂ps !== nothing &&
                      !iszero(ComponentArray(∂ps |> cpu_device())) &&
                      all(isfinite, ComponentArray(∂ps |> cpu_device()))

                ongpu && continue

                ∂x_fd = FiniteDifferences.grad(
                    central_fdm(5, 1), x -> loss_fn(model, x, ps, st), X)[1]
                ∂ps_fd = FiniteDifferences.grad(
                    central_fdm(5, 1), p -> loss_fn(model, X, p, st), ps)[1]

                @test ∂x≈∂x_fd rtol=1e-3 atol=1e-3
                @test check_approx(∂ps, ∂ps_fd; rtol=1e-3, atol=1e-3)
            end
        end

        @testset "Structured Matrix: Issue LuxDL/Lux.jl#602" begin
            model = @compact(; potential=Dense(5 => 5, gelu)) do x
                return reshape(diag(only(Zygote.jacobian(potential, x))), size(x))
            end

            ps, st = Lux.setup(rng, model) |> dev
            x = randn(rng, Float32, 5, 5) |> aType

            ∂x, ∂ps, _ = Zygote.gradient(Base.Fix1(sum, abs2) ∘ first ∘ model, x, ps, st)

            ps_cpu = ps |> cpu_device()
            st_cpu = st |> cpu_device()
            x_cpu = x |> cpu_device()

            # Use FiniteDiff on CPU
            ∂x_fd = FiniteDifferences.grad(
                central_fdm(5, 1), x -> sum(abs2, first(model(x, ps_cpu, st_cpu))), x_cpu)[1]
            ∂ps_fd = FiniteDifferences.grad(
                central_fdm(5, 1), p -> sum(abs2, first(model(x_cpu, p, st_cpu))), ps_cpu)[1]

            ∂x_cpu = ∂x |> cpu_device()
            ∂ps_cpu = ∂ps |> cpu_device()

            @test ∂x_cpu≈∂x_fd rtol=1e-3 atol=1e-3
            @test check_approx(∂ps_cpu, ∂ps_fd; rtol=1e-3, atol=1e-3)
        end
    end
end

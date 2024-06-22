@testitem "Nested AD: Input Gradient/Jacobian" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ComponentArrays, FiniteDifferences, ForwardDiff, LinearAlgebra, Zygote

    Base.isfinite(::Nothing) = true

    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # FIXME: AMDGPU takes too long right now
        mode === "amdgpu" && continue

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

            # smodel | ForwardDiff.jacobian
            loss_function1 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer{true}(model, ps, st)
                return sum(abs2, ForwardDiff.jacobian(smodel, x))
            end

            # smodel | Zygote.jacobian
            loss_function2 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer{true}(model, ps, st)
                return sum(abs2, only(Zygote.jacobian(smodel, x)))
            end

            # sum(abs2) ∘ smodel | ForwardDiff.gradient
            loss_function3 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer{true}(model, ps, st)
                return sum(abs2, ForwardDiff.gradient(Base.Fix1(sum, abs2) ∘ smodel, x))
            end

            # sum(abs2) ∘ smodel | Zygote.gradient
            loss_function4 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer{true}(model, ps, st)
                return sum(abs2, only(Zygote.gradient(Base.Fix1(sum, abs2) ∘ smodel, x)))
            end

            loss_fns = ongpu ? (loss_function1, loss_function2, loss_function4) :
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
    end
end

@testitem "Nested AD: Parameter Gradient/Jacobian" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ComponentArrays, FiniteDifferences, ForwardDiff, LinearAlgebra, Zygote

    Base.isfinite(::Nothing) = true

    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # FIXME: AMDGPU takes too long right now
        mode === "amdgpu" && continue

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
            ps, st = Lux.setup(rng, model)
            ps = ps |> ComponentArray |> dev
            st = st |> dev

            # smodel | ForwardDiff.jacobian
            loss_function1 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer{true}(model, ps, st)
                return sum(abs2, ForwardDiff.jacobian(Base.Fix1(smodel, x), ps))
            end

            # smodel | Zygote.jacobian
            loss_function2 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer{true}(model, ps, st)
                return sum(abs2, only(Zygote.jacobian(Base.Fix1(smodel, x), ps)))
            end

            # sum(abs2) ∘ smodel | ForwardDiff.gradient
            loss_function3 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer{true}(model, ps, st)
                return sum(abs2,
                    ForwardDiff.gradient(Base.Fix1(sum, abs2) ∘ Base.Fix1(smodel, x), ps))
            end

            # sum(abs2) ∘ smodel | Zygote.gradient
            loss_function4 = (model, x, ps, st) -> begin
                smodel = StatefulLuxLayer{true}(model, ps, st)
                return sum(abs2,
                    only(Zygote.gradient(Base.Fix1(sum, abs2) ∘ Base.Fix1(smodel, x), ps)))
            end

            loss_fns = ongpu ? (loss_function1, loss_function2, loss_function4) :
                       (loss_function1, loss_function2, loss_function3, loss_function4)

            for loss_fn in loss_fns
                @test_nowarn loss_fn(model, X, ps, st)
                @test loss_fn(model, X, ps, st) isa Number

                _, ∂x, ∂ps, _ = Zygote.gradient(loss_fn, model, X, ps, st)

                @test ∂x !== nothing && !iszero(∂x) && all(isfinite, ∂x)
                @test ∂ps !== nothing &&
                      !iszero(∂ps |> cpu_device()) &&
                      all(isfinite, ∂ps |> cpu_device())

                ongpu && continue

                ∂x_fd = FiniteDifferences.grad(
                    central_fdm(5, 1), x -> loss_fn(model, x, ps, st), X)[1]
                ∂ps_fd = FiniteDifferences.grad(
                    central_fdm(5, 1), p -> loss_fn(model, X, p, st), ps)[1]

                @test ∂x≈∂x_fd rtol=1e-3 atol=1e-3
                @test check_approx(∂ps, ∂ps_fd; rtol=1e-3, atol=1e-3)
            end
        end
    end
end

@testitem "Nested AD: Structured Matrix LuxDL/Lux.jl#602" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ComponentArrays, FiniteDifferences, ForwardDiff, LinearAlgebra, Zygote

    Base.isfinite(::Nothing) = true

    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # FIXME: AMDGPU takes too long right now
        mode === "amdgpu" && continue

        @testset "Structured Matrix: Issue LuxDL/Lux.jl#602" begin
            model = @compact(; potential=Dense(5 => 5, gelu)) do x
                @return reshape(diag(only(Zygote.jacobian(potential, x))), size(x))
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

@testitem "Nested AD: VJP & JVP" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ComponentArrays, FiniteDifferences, ForwardDiff, LinearAlgebra, Zygote, ADTypes

    Base.isfinite(::Nothing) = true

    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # FIXME: AMDGPU takes too long right now
        mode === "amdgpu" && continue

        models = (
            Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()), BatchNorm(4),
                Conv((3, 3), 4 => 1, gelu; pad=SamePad())),
            Chain(Dense(2, 4, gelu), Dense(4, 1)))
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (model, X) in zip(models, Xs)
            ps, st = Lux.setup(rng, model) |> dev

            vjp_input = first(model(X, ps, st))
            jvp_input = aType(randn(rng, Float32, size(X)...))

            function loss_function_vjp(model, X, ps, st, vjp_input)
                smodel = StatefulLuxLayer{true}(model, ps, st)
                vjp = vector_jacobian_product(smodel, AutoZygote(), X, vjp_input)
                return sum(vjp)
            end

            function loss_function_vjp_jacobian(model, X, ps, st, vjp_input)
                smodel = StatefulLuxLayer{true}(model, ps, st)
                J = only(Zygote.jacobian(smodel, X))
                return sum(J' * vec(vjp_input))
            end

            function loss_function_jvp(model, X, ps, st, jvp_input)
                smodel = StatefulLuxLayer{true}(model, ps, st)
                jvp = jacobian_vector_product(smodel, AutoForwardDiff(), X, jvp_input)
                return sum(jvp)
            end

            function loss_function_jvp_jacobian(model, X, ps, st, jvp_input)
                smodel = StatefulLuxLayer{true}(model, ps, st)
                J = only(Zygote.jacobian(smodel, X))
                return sum(J * vec(jvp_input))
            end

            @test_nowarn loss_function_vjp(model, X, ps, st, vjp_input)
            @test loss_function_vjp(model, X, ps, st, vjp_input) isa Number
            @test loss_function_vjp(model, X, ps, st, vjp_input) ≈
                  loss_function_vjp_jacobian(model, X, ps, st, vjp_input)

            _, ∂x, ∂ps, _ = Zygote.gradient(loss_function_vjp, model, X, ps, st, vjp_input)
            _, ∂x_vjp, ∂ps_vjp, _, _ = Zygote.gradient(
                loss_function_vjp_jacobian, model, X, ps, st, vjp_input)

            @test ∂x≈∂x_vjp rtol=1e-3 atol=1e-3
            @test check_approx(∂ps, ∂ps_vjp; rtol=1e-3, atol=1e-3)

            @test_nowarn loss_function_jvp(model, X, ps, st, jvp_input)
            @test loss_function_jvp(model, X, ps, st, jvp_input) isa Number
            @test loss_function_jvp(model, X, ps, st, jvp_input) ≈
                  loss_function_jvp_jacobian(model, X, ps, st, jvp_input)

            _, ∂x, ∂ps, _ = Zygote.gradient(loss_function_jvp, model, X, ps, st, jvp_input)
            _, ∂x_jvp, ∂ps_jvp, _, _ = Zygote.gradient(
                loss_function_jvp_jacobian, model, X, ps, st, jvp_input)

            @test ∂x≈∂x_jvp rtol=1e-3 atol=1e-3
            @test check_approx(∂ps, ∂ps_jvp; rtol=1e-3, atol=1e-3)
        end
    end
end

@testitem "VJP/JVP Interface Test" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ForwardDiff, Functors, Zygote, ADTypes

    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # FIXME: AMDGPU takes too long right now
        mode === "amdgpu" && continue

        x = rand(rng, 3, 3) |> aType
        v = vec(rand(rng, 3, 3)) |> aType

        ftest(x) = abs2.(x)

        J = ForwardDiff.jacobian(ftest, x)
        Jv = J * v
        vJ = vec(v' * J)

        Jv_fdiff = vec(jacobian_vector_product(ftest, AutoForwardDiff(), x, v))
        @test Jv≈Jv_fdiff rtol=1e-3 atol=1e-3

        vJ_zyg = vec(vector_jacobian_product(ftest, AutoZygote(), x, reshape(v, size(x))))
        @test vJ≈vJ_zyg rtol=1e-3 atol=1e-3
    end

    struct functorABC{A, B}
        a::A
        b::B
    end

    @functor functorABC

    function ftest(st)
        return functorABC(st.functor.a .* st.functor.b, st.tup[1] .* st.tup[2])
    end

    nt = (functor=functorABC(rand(rng, 3), rand(rng, 3)), tup=(rand(rng, 3), rand(rng, 3)))
    u = (functor=functorABC(rand(rng, 3), rand(rng, 3)), tup=(rand(rng, 3), rand(rng, 3)))

    @test_nowarn jacobian_vector_product(ftest, AutoForwardDiff(), nt, u)
end

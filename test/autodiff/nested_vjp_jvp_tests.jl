using StableRNGs, Lux, ForwardDiff, Zygote, Enzyme, ComponentArrays, Functors, ADTypes
using LuxTestUtils, Test
using DispatchDoctor: allow_unstable

include("../shared_testsetup.jl")

@testset "Nested AD: VJP & JVP" begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        models = (
            Chain(
                Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
                BatchNorm(4),
                Conv((3, 3), 4 => 1, gelu; pad=SamePad()),
            ),
            Chain(Dense(2, 4, gelu), Dense(4, 1)),
        )
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (model, X) in zip(models, Xs)
            ps, st = dev(Lux.setup(rng, model))

            vjp_input = first(model(X, ps, st))
            jvp_input = aType(randn(rng, Float32, size(X)...))

            function loss_function_vjp(model, X, ps, st, vjp_input)
                smodel = StatefulLuxLayer(model, ps, st)
                vjp = vector_jacobian_product(smodel, AutoZygote(), X, vjp_input)
                return sum(vjp)
            end

            function loss_function_vjp_jacobian(model, X, ps, st, vjp_input)
                smodel = StatefulLuxLayer(model, ps, st)
                J = only(Zygote.jacobian(smodel, X))
                return sum(J' * vec(vjp_input))
            end

            function loss_function_jvp(model, X, ps, st, jvp_input)
                smodel = StatefulLuxLayer(model, ps, st)
                jvp = jacobian_vector_product(smodel, AutoForwardDiff(), X, jvp_input)
                return sum(jvp)
            end

            function loss_function_jvp_jacobian(model, X, ps, st, jvp_input)
                smodel = StatefulLuxLayer(model, ps, st)
                J = only(Zygote.jacobian(smodel, X))
                return sum(J * vec(jvp_input))
            end

            @test loss_function_vjp(model, X, ps, st, vjp_input) isa Number
            @test loss_function_vjp(model, X, ps, st, vjp_input) ≈
                loss_function_vjp_jacobian(model, X, ps, st, vjp_input)

            _, ∂x, ∂ps, _ = allow_unstable() do
                Zygote.gradient(loss_function_vjp, model, X, ps, st, vjp_input)
            end
            _, ∂x_vjp, ∂ps_vjp, _, _ = allow_unstable() do
                Zygote.gradient(loss_function_vjp_jacobian, model, X, ps, st, vjp_input)
            end

            @test ∂x ≈ ∂x_vjp rtol = 1.0e-3 atol = 1.0e-3
            @test check_approx(∂ps, ∂ps_vjp; rtol=1.0e-3, atol=1.0e-3)

            @test loss_function_jvp(model, X, ps, st, jvp_input) isa Number
            @test loss_function_jvp(model, X, ps, st, jvp_input) ≈
                loss_function_jvp_jacobian(model, X, ps, st, jvp_input)

            _, ∂x, ∂ps, _ = allow_unstable() do
                Zygote.gradient(loss_function_jvp, model, X, ps, st, jvp_input)
            end
            _, ∂x_jvp, ∂ps_jvp, _, _ = allow_unstable() do
                Zygote.gradient(loss_function_jvp_jacobian, model, X, ps, st, jvp_input)
            end

            @test ∂x ≈ ∂x_jvp rtol = 1.0e-3 atol = 1.0e-3
            @test check_approx(∂ps, ∂ps_jvp; rtol=1.0e-3, atol=1.0e-3)
        end
    end
end

@testset "VJP/JVP Interface Test" begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = aType(rand(rng, 3, 3))
        v = aType(vec(rand(rng, 3, 3)))

        ftest(x) = abs2.(x)

        J = ForwardDiff.jacobian(ftest, x)
        Jv = J * v
        vJ = vec(v' * J)

        Jv_fdiff = vec(
            allow_unstable() do
                jacobian_vector_product(ftest, AutoForwardDiff(), x, v)
            end,
        )
        @test Jv ≈ Jv_fdiff rtol = 1.0e-3 atol = 1.0e-3

        vJ_zyg = vec(
            allow_unstable() do
                vector_jacobian_product(ftest, AutoZygote(), x, reshape(v, size(x)))
            end,
        )
        @test vJ ≈ vJ_zyg rtol = 1.0e-3 atol = 1.0e-3
    end

    struct functorABC{A,B}
        a::A
        b::B
    end

    @functor functorABC

    function ftest(st)
        return functorABC(st.functor.a .* st.functor.b, st.tup[1] .* st.tup[2])
    end

    rng = StableRNG(1234)
    nt = (functor=functorABC(rand(rng, 3), rand(rng, 3)), tup=(rand(rng, 3), rand(rng, 3)))
    u = (functor=functorABC(rand(rng, 3), rand(rng, 3)), tup=(rand(rng, 3), rand(rng, 3)))

    @test jacobian_vector_product(ftest, AutoForwardDiff(), nt, u) isa Any
end

@testset "VJP on StatefulLuxLayer and ComposedFunction" begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Chain(Dense(2, 4, gelu), Dense(4, 3))
        ps, st = dev(Lux.setup(rng, model))
        smodel = StatefulLuxLayer(model, ps, st)
        X = aType(randn(rng, Float32, 2, 4))
        u = first(model(X, ps, st))

        # StatefulLuxLayer VJP with AutoZygote and AutoEnzyme
        vjp_z = vector_jacobian_product(smodel, AutoZygote(), X, u)
        vjp_e = vector_jacobian_product(smodel, AutoEnzyme(), X, u)
        @test vjp_z ≈ vjp_e rtol = 1.0e-3 atol = 1.0e-3

        # Compare with Zygote Jacobian
        J = only(Zygote.jacobian(smodel, X))
        @test vec(vjp_z) ≈ vec(J' * vec(u)) rtol = 1.0e-3 atol = 1.0e-3

        # ComposedFunction VJP
        f_composed = (x -> x .* 2.0f0) ∘ smodel
        u_c = f_composed(X)
        vjp_c_z = vector_jacobian_product(f_composed, AutoZygote(), X, u_c)
        vjp_c_e = vector_jacobian_product(
            f_composed,
            AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)),
            X,
            u_c,
        )
        @test vjp_c_z ≈ vjp_c_e rtol = 1.0e-3 atol = 1.0e-3

        # Parameter VJP: Base.Fix1(smodel, X)
        f_fix1 = Base.Fix1(smodel, X)
        vjp_fix1_z = vector_jacobian_product(f_fix1, AutoZygote(), ps, u)
        @test vjp_fix1_z isa NamedTuple
        @test keys(vjp_fix1_z) == keys(ps)

        # Nested AD through vector_jacobian_product with StatefulLuxLayer
        function loss_vjp(model, X, ps, st, u)
            return sum(
                vector_jacobian_product(StatefulLuxLayer(model, ps, st), AutoZygote(), X, u)
            )
        end
        _, ∇x, ∇ps, _, _ = allow_unstable() do
            Zygote.gradient(loss_vjp, model, X, ps, st, u)
        end
        @test ∇x !== nothing
        @test ∇ps !== nothing

        # Nested AD through vector_jacobian_product with ComposedFunction
        function loss_comp_vjp(model, X, ps, st, u_c)
            return sum(
                vector_jacobian_product(
                    (x -> x .* 2.0f0) ∘ StatefulLuxLayer(model, ps, st),
                    AutoZygote(),
                    X,
                    u_c,
                ),
            )
        end
        _, ∇x_c, ∇ps_c, _, _ = allow_unstable() do
            Zygote.gradient(loss_comp_vjp, model, X, ps, st, u_c)
        end
        @test ∇x_c !== nothing
        @test ∇ps_c !== nothing
    end
end

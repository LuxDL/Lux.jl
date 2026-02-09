using StableRNGs, Lux, ForwardDiff, Zygote, ComponentArrays, Functors, ADTypes
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
            model = maybe_rewrite_to_crosscor(mode, model)
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

@testsetup module NestedADInputTestSetup

using StableRNGs, Lux, ForwardDiff, Zygote, ComponentArrays
using LuxTestUtils, Test
using DispatchDoctor: allow_unstable

rng = StableRNG(1234)

function test_nested_ad_input_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
    loss_fn === loss_function3 && ongpu && return

    ps, st = Lux.setup(rng, model) |> dev
    X = aType(X)

    l = allow_unstable() do
        loss_fn(model, X, ps, st)
    end
    @test l isa Number
    @test isfinite(l) && !isnan(l)

    _, ∂x, ∂ps, _ = allow_unstable() do
        Zygote.gradient(loss_fn, model, X, ps, st)
    end

    @test ∂x !== nothing && !iszero(∂x) && all(x -> x === nothing || isfinite(x), ∂x)
    @test ∂ps !== nothing &&
          !iszero(ComponentArray(∂ps |> cpu_device())) &&
          all(x -> x === nothing || isfinite(x), ComponentArray(∂ps |> cpu_device()))

    allow_unstable() do
        @test_gradients((x, ps)->loss_fn(model, x, ps, st), X, ps;
            atol=1.0f-3, rtol=1.0f-1,
            soft_fail=[AutoFiniteDiff()],
            skip_backends=[AutoReverseDiff(), AutoTracker(), AutoEnzyme()])
    end
end

const Xs = (randn(rng, Float32, 3, 3, 2, 2), randn(rng, Float32, 2, 2),
    randn(rng, Float32, 2, 2), randn(rng, Float32, 3, 3, 2, 2))

const models = (
    Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()), BatchNorm(4),
        Conv((3, 3), 4 => 2, gelu; pad=SamePad()),
        BatchNorm(2), FlattenLayer(), Dense(18 => 2)),
    Chain(Dense(2, 4), GroupNorm(4, 2, gelu), Dense(4, 2)),
    Chain(Dense(2, 4), BatchNorm(4, gelu), Dense(4, 2)),
    Chain(Conv((3, 3), 2 => 3, gelu; pad=SamePad()), BatchNorm(3),
        Conv((3, 3), 3 => 2, gelu; pad=SamePad()),
        BatchNorm(2), FlattenLayer(), Dense(18 => 1)))

# smodel | ForwardDiff.jacobian
function loss_function1(model, x, ps, st)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    return sum(abs2, ForwardDiff.jacobian(smodel, x) .* 0.01f0)
end

# smodel | Zygote.jacobian
function loss_function2(model, x, ps, st)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    return sum(abs2, only(Zygote.jacobian(smodel, x)) .* 0.01f0)
end

# sum(abs2) ∘ smodel | ForwardDiff.gradient
function loss_function3(model, x, ps, st)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    return sum(abs2, ForwardDiff.gradient(Base.Fix1(sum, abs2) ∘ smodel, x) .* 0.01f0)
end

# sum(abs2) ∘ smodel | Zygote.gradient
function loss_function4(model, x, ps, st)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    return sum(abs2, only(Zygote.gradient(Base.Fix1(sum, abs2) ∘ smodel, x)) .* 0.01f0)
end

const ALL_TEST_CONFIGS = Iterators.product(
    zip(Xs, models), (loss_function1, loss_function2, loss_function3, loss_function4))

const TEST_BLOCKS = collect(Iterators.partition(
    ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 4)))

export test_nested_ad_input_gradient_jacobian, TEST_BLOCKS

end

@testitem "Nested AD: Input Gradient/Jacobian Group 1" setup=[
    SharedTestSetup, NestedADInputTestSetup] tags=[:autodiff] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(summary(X)) $(loss_fn)" for ((X, model), loss_fn) in TEST_BLOCKS[1]
            model = maybe_rewrite_to_crosscor(mode, model)
            test_nested_ad_input_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
        end
    end
end

@testitem "Nested AD: Input Gradient/Jacobian Group 2" setup=[
    SharedTestSetup, NestedADInputTestSetup] tags=[:autodiff] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(summary(X)) $(loss_fn)" for ((X, model), loss_fn) in TEST_BLOCKS[2]
            model = maybe_rewrite_to_crosscor(mode, model)
            test_nested_ad_input_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
        end
    end
end

@testitem "Nested AD: Input Gradient/Jacobian Group 3" setup=[
    SharedTestSetup, NestedADInputTestSetup] tags=[:autodiff] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(summary(X)) $(loss_fn)" for ((X, model), loss_fn) in TEST_BLOCKS[3]
            model = maybe_rewrite_to_crosscor(mode, model)
            test_nested_ad_input_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
        end
    end
end

@testitem "Nested AD: Input Gradient/Jacobian Group 4" setup=[
    SharedTestSetup, NestedADInputTestSetup] tags=[:autodiff] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(summary(X)) $(loss_fn)" for ((X, model), loss_fn) in TEST_BLOCKS[4]
            model = maybe_rewrite_to_crosscor(mode, model)
            test_nested_ad_input_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
        end
    end
end

@testsetup module NestedADParameterTestSetup

using StableRNGs, Lux, ForwardDiff, Zygote, ComponentArrays
using LuxTestUtils, Test
using DispatchDoctor: allow_unstable

rng = StableRNG(1234)

function test_nested_ad_parameter_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
    loss_fn === loss_function3 && ongpu && return

    ps, st = Lux.setup(rng, model)
    ps = ps |> ComponentArray |> dev
    st = st |> dev
    X = aType(X)

    l = allow_unstable() do
        loss_fn(model, X, ps, st)
    end
    @test l isa Number
    @test isfinite(l) && !isnan(l)

    _, ∂x, ∂ps, _ = allow_unstable() do
        Zygote.gradient(loss_fn, model, X, ps, st)
    end

    @test ∂x !== nothing && !iszero(∂x) && all(x -> x === nothing || isfinite(x), ∂x)
    @test ∂ps !== nothing &&
          !iszero(ComponentArray(∂ps |> cpu_device())) &&
          all(x -> x === nothing || isfinite(x), ComponentArray(∂ps |> cpu_device()))

    allow_unstable() do
        @test_gradients((x, ps)->loss_fn(model, x, ps, st), X, ps;
            atol=1.0f-3, rtol=1.0f-1,
            soft_fail=[AutoFiniteDiff()],
            skip_backends=[AutoReverseDiff(), AutoTracker(), AutoEnzyme()])
    end
end

const Xs = (randn(rng, Float32, 3, 3, 2, 2), randn(rng, Float32, 2, 2),
    randn(rng, Float32, 2, 2), randn(rng, Float32, 3, 3, 2, 2))

const models = (
    Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()), BatchNorm(4),
        Conv((3, 3), 4 => 2, gelu; pad=SamePad()),
        BatchNorm(2), FlattenLayer(), Dense(18 => 2)),
    Chain(Dense(2, 4, gelu), Dense(4, 2)),
    Chain(Dense(2, 4, gelu), BatchNorm(4, sigmoid), Dense(4, 2)),
    Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()), BatchNorm(4),
        Conv((3, 3), 4 => 2, tanh; pad=SamePad()),
        BatchNorm(2), FlattenLayer(), Dense(18 => 1)))

# smodel | ForwardDiff.jacobian
function loss_function1(model, x, ps, st)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    return sum(abs2, ForwardDiff.jacobian(Base.Fix1(smodel, x), ps) .* 0.01f0)
end

# smodel | Zygote.jacobian
function loss_function2(model, x, ps, st)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    return sum(abs2, only(Zygote.jacobian(Base.Fix1(smodel, x), ps)) .* 0.01f0)
end

# sum(abs2) ∘ smodel | ForwardDiff.gradient
function loss_function3(model, x, ps, st)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    return sum(abs2,
        ForwardDiff.gradient(Base.Fix1(sum, abs2) ∘ Base.Fix1(smodel, x), ps) .* 0.01f0)
end

# sum(abs2) ∘ smodel | Zygote.gradient
function loss_function4(model, x, ps, st)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    return sum(abs2,
        only(Zygote.gradient(Base.Fix1(sum, abs2) ∘ Base.Fix1(smodel, x), ps)) .* 0.01f0)
end

const ALL_TEST_CONFIGS = Iterators.product(
    zip(Xs, models), (loss_function1, loss_function2, loss_function3, loss_function4))

const TEST_BLOCKS = collect(Iterators.partition(
    ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 4)))

export test_nested_ad_parameter_gradient_jacobian, TEST_BLOCKS

end

@testitem "Nested AD: Parameter Gradient/Jacobian Group 1" setup=[
    SharedTestSetup, NestedADParameterTestSetup] tags=[:autodiff] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(summary(X)) $(loss_fn)" for ((X, model), loss_fn) in TEST_BLOCKS[1]
            model = maybe_rewrite_to_crosscor(mode, model)
            test_nested_ad_parameter_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
        end
    end
end

@testitem "Nested AD: Parameter Gradient/Jacobian Group 2" setup=[
    SharedTestSetup, NestedADParameterTestSetup] tags=[:autodiff] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(summary(X)) $(loss_fn)" for ((X, model), loss_fn) in TEST_BLOCKS[2]
            model = maybe_rewrite_to_crosscor(mode, model)
            test_nested_ad_parameter_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
        end
    end
end

@testitem "Nested AD: Parameter Gradient/Jacobian Group 3" setup=[
    SharedTestSetup, NestedADParameterTestSetup] tags=[:autodiff] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(summary(X)) $(loss_fn)" for ((X, model), loss_fn) in TEST_BLOCKS[3]
            model = maybe_rewrite_to_crosscor(mode, model)
            test_nested_ad_parameter_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
        end
    end
end

@testitem "Nested AD: Parameter Gradient/Jacobian Group 4" setup=[
    SharedTestSetup, NestedADParameterTestSetup] tags=[:autodiff] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(summary(X)) $(loss_fn)" for ((X, model), loss_fn) in TEST_BLOCKS[4]
            model = maybe_rewrite_to_crosscor(mode, model)
            test_nested_ad_parameter_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
        end
    end
end

@testitem "Nested AD: Structured Matrix LuxDL/Lux.jl#602" setup=[SharedTestSetup] tags=[:autodiff] begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Structured Matrix: Issue LuxDL/Lux.jl#602" begin
            model = @compact(; potential=Dense(5 => 5, gelu)) do x
                @return reshape(diag(only(Zygote.jacobian(potential, x))), size(x))
            end

            ps, st = Lux.setup(rng, model) |> dev
            x = randn(rng, Float32, 5, 5) |> aType

            __f = let model = model, st = st
                (x, ps) -> sum(abs2, first(model(x, ps, st)))
            end

            @test_gradients(__f, x,
                ps; atol=1.0f-3,
                rtol=1.0f-3, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoEnzyme()])
        end
    end
end

@testitem "Nested AD: VJP & JVP" setup=[SharedTestSetup] tags=[:autodiff] begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        models = (
            Chain(Conv((3, 3), 2 => 4, gelu; pad=SamePad()), BatchNorm(4),
                Conv((3, 3), 4 => 1, gelu; pad=SamePad())),
            Chain(Dense(2, 4, gelu), Dense(4, 1)))
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (model, X) in zip(models, Xs)
            model = maybe_rewrite_to_crosscor(mode, model)
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

            @test loss_function_vjp(model, X, ps, st, vjp_input) isa Number
            @test loss_function_vjp(model, X, ps, st, vjp_input) ≈
                  loss_function_vjp_jacobian(model, X, ps, st, vjp_input)

            _, ∂x, ∂ps, _ = allow_unstable() do
                Zygote.gradient(loss_function_vjp, model, X, ps, st, vjp_input)
            end
            _, ∂x_vjp, ∂ps_vjp, _, _ = allow_unstable() do
                Zygote.gradient(loss_function_vjp_jacobian, model, X, ps, st, vjp_input)
            end

            @test ∂x≈∂x_vjp rtol=1e-3 atol=1e-3
            @test check_approx(∂ps, ∂ps_vjp; rtol=1e-3, atol=1e-3)

            @test loss_function_jvp(model, X, ps, st, jvp_input) isa Number
            @test loss_function_jvp(model, X, ps, st, jvp_input) ≈
                  loss_function_jvp_jacobian(model, X, ps, st, jvp_input)

            _, ∂x, ∂ps, _ = allow_unstable() do
                Zygote.gradient(loss_function_jvp, model, X, ps, st, jvp_input)
            end
            _, ∂x_jvp, ∂ps_jvp, _, _ = allow_unstable() do
                Zygote.gradient(loss_function_jvp_jacobian, model, X, ps, st, jvp_input)
            end

            @test ∂x≈∂x_jvp rtol=1e-3 atol=1e-3
            @test check_approx(∂ps, ∂ps_jvp; rtol=1e-3, atol=1e-3)
        end
    end
end

@testitem "VJP/JVP Interface Test" setup=[SharedTestSetup] tags=[:autodiff] begin
    using Functors, ADTypes

    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = rand(rng, 3, 3) |> aType
        v = vec(rand(rng, 3, 3)) |> aType

        ftest(x) = abs2.(x)

        J = ForwardDiff.jacobian(ftest, x)
        Jv = J * v
        vJ = vec(v' * J)

        Jv_fdiff = vec(allow_unstable() do
            jacobian_vector_product(ftest, AutoForwardDiff(), x, v)
        end)
        @test Jv≈Jv_fdiff rtol=1e-3 atol=1e-3

        vJ_zyg = vec(allow_unstable() do
            vector_jacobian_product(ftest, AutoZygote(), x, reshape(v, size(x)))
        end)
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

    @test jacobian_vector_product(ftest, AutoForwardDiff(), nt, u) isa Any
end

@testitem "Nested AD: Issue #743 (eval + gradient)" setup=[SharedTestSetup] tags=[:autodiff] begin
    function loss_function(model, ps, st, x)
        smodel = StatefulLuxLayer{true}(model, ps, st)
        y_pred = smodel(x)
        dy_pred = only(Zygote.gradient(sum ∘ smodel, x))
        loss = sum(dy_pred .+ y_pred .^ 2 / 2)
        return loss
    end

    rng = StableRNG(1234)
    model = Chain(Dense(1 => 8, sigmoid), Dense(8 => 1))
    ps, st = Lux.setup(rng, model)
    x = randn(rng, Float32, 1, 12)

    __f = let model = model, st = st
        (x, ps) -> loss_function(model, ps, st, x)
    end

    @test_gradients(__f, x, ps; atol=1.0f-3,
        rtol=1.0f-3, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoEnzyme()])
end

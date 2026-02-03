using StableRNGs, Lux, ForwardDiff, Zygote, ComponentArrays, Functors, ADTypes
using LuxTestUtils, Test
using DispatchDoctor: allow_unstable

include("../shared_testsetup.jl")

rng_nested = StableRNG(1234)

function test_nested_ad_parameter_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
    loss_fn === loss_function_param3 && ongpu && return nothing

    ps, st = Lux.setup(rng_nested, model)
    ps = dev(ComponentArray(ps))
    st = dev(st)
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
        !iszero(ComponentArray(cpu_device()(∂ps))) &&
        all(x -> x === nothing || isfinite(x), ComponentArray(cpu_device()(∂ps)))

    allow_unstable() do
        @test_gradients(
            loss_fn,
            Constant(model),
            X,
            ps,
            Constant(st);
            atol=1.0f-3,
            rtol=1.0f-1,
            soft_fail=[AutoFiniteDiff()],
            skip_backends=[AutoEnzyme()]
        )
    end
end

const PARAM_Xs = (randn(rng_nested, Float32, 2, 2), randn(rng_nested, Float32, 3, 3, 2, 2))

const PARAM_models = (
    Chain(Dense(2, 4, gelu), BatchNorm(4, sigmoid), Dense(4, 2)),
    Chain(
        Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
        BatchNorm(4),
        Conv((3, 3), 4 => 2, tanh; pad=SamePad()),
        BatchNorm(2),
        FlattenLayer(),
        Dense(18 => 1),
    ),
)

# smodel | ForwardDiff.jacobian
function loss_function_param1(model, x, ps, st)
    smodel = StatefulLuxLayer(model, ps, st)
    return sum(abs2, ForwardDiff.jacobian(Base.Fix1(smodel, x), ps) .* 0.01f0)
end

# smodel | Zygote.jacobian
function loss_function_param2(model, x, ps, st)
    smodel = StatefulLuxLayer(model, ps, st)
    return sum(abs2, only(Zygote.jacobian(Base.Fix1(smodel, x), ps)) .* 0.01f0)
end

# sum(abs2) ∘ smodel | ForwardDiff.gradient
function loss_function_param3(model, x, ps, st)
    smodel = StatefulLuxLayer(model, ps, st)
    return sum(
        abs2,
        ForwardDiff.gradient(Base.Fix1(sum, abs2) ∘ Base.Fix1(smodel, x), ps) .* 0.01f0,
    )
end

# sum(abs2) ∘ smodel | Zygote.gradient
function loss_function_param4(model, x, ps, st)
    smodel = StatefulLuxLayer(model, ps, st)
    return sum(
        abs2,
        only(Zygote.gradient(Base.Fix1(sum, abs2) ∘ Base.Fix1(smodel, x), ps)) .* 0.01f0,
    )
end

const PARAM_ALL_TEST_CONFIGS = Iterators.product(
    zip(PARAM_Xs, PARAM_models),
    (
        loss_function_param1,
        loss_function_param2,
        loss_function_param3,
        loss_function_param4,
    ),
)

@testset "Nested AD: Parameter Gradient/Jacobian" begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset "$(summary(X)) $(loss_fn)" for ((X, model), loss_fn) in
                                                PARAM_ALL_TEST_CONFIGS
            model = maybe_rewrite_to_crosscor(mode, model)
            test_nested_ad_parameter_gradient_jacobian(aType, dev, ongpu, loss_fn, X, model)
        end
    end
end

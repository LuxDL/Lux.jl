include("../shared_testsetup.jl")

using LuxLib, Test, StableRNGs, NNlib, LuxTestUtils
using ReverseDiff, Tracker

bias_act_loss1(act, x, b) = sum(abs2, act.(x .+ LuxLib.Impl.reshape_bias(x, b)))
bias_act_loss2(act, x, b) = sum(abs2, bias_activation(act, x, b))
bias_act_loss3(act, x, b) = sum(abs2, bias_activation!!(act, copy(x), b))

struct __Fix1{F,A}
    f::F
    act::A
end
(f::__Fix1)(x, b) = f.f(f.act, x, b)

const ALL_ACTIVATIONS = [
    identity,
    relu,
    sigmoid,
    sigmoid_fast,
    softplus,
    logsigmoid,
    gelu,
    swish,
    lisht,
    tanh,
    tanh_fast,
]

const ALL_TEST_CONFIGS = vcat(
    [(act, Float32, (2, 2, 3, 4)) for act in ALL_ACTIVATIONS],
    [(act, Float32, (4, 5)) for act in (identity, relu, sigmoid, gelu)],
    [
        (identity, Float64, (2, 2, 3, 4)),
        (gelu, Float64, (2, 2, 3, 4)),
        (relu, Float64, (4, 5)),
        (sigmoid, Float64, (4, 5)),
    ],
)

@testset "Bias Activation" begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "$act, $T, $sz" for (act, T, sz) in ALL_TEST_CONFIGS
            !fp64 && T == Float64 && continue

            x = aType(rand(rng, T, sz))
            b = aType(rand(rng, T, sz[end - 1]))

            y1 = bias_act_loss1(act, x, b)
            y2 = bias_act_loss2(act, x, b)
            y3 = bias_act_loss3(act, x, b)

            fp16 = T == Float16
            atol = fp16 ? 1.0f-2 : 1.0f-3
            rtol = fp16 ? 1.0f-2 : 1.0f-3

            @test y1 ≈ y2 atol = atol rtol = rtol
            @test y1 ≈ y3 atol = atol rtol = rtol
            @test eltype(y1) == T
            @test eltype(y2) == T
            @test eltype(y3) == T

            @test @inferred(bias_act_loss1(act, x, b)) isa Any
            @test @inferred(bias_act_loss2(act, x, b)) isa Any
            @test @inferred(bias_act_loss3(act, x, b)) isa Any

            @jet bias_act_loss2(act, x, b)
            @jet bias_act_loss3(act, x, b)

            @test_gradients(
                __Fix1(bias_act_loss1, act),
                x,
                b;
                atol,
                rtol,
                soft_fail=fp16 ? [AutoFiniteDiff()] : []
            )
            @test_gradients(
                __Fix1(bias_act_loss2, act),
                x,
                b;
                atol,
                rtol,
                soft_fail=fp16 ? [AutoFiniteDiff()] : []
            )
            @test_gradients(
                __Fix1(bias_act_loss3, act),
                x,
                b;
                atol,
                rtol,
                soft_fail=fp16 ? [AutoFiniteDiff()] : []
            )
        end
    end
end

@testset "Bias Activation (ReverseDiff)" begin
    x = rand(Float32, 3, 4)
    b = rand(Float32, 3)
    act = tanh

    z = bias_activation(act, ReverseDiff.track(x), b)
    @test z isa ReverseDiff.TrackedArray  # If this fails then we fail to compile the tape

    z = bias_activation(identity, ReverseDiff.track(x), b)
    @test z isa ReverseDiff.TrackedArray

    z = bias_activation(act, Tracker.param(x), b)
    @test z isa Tracker.TrackedArray

    z = bias_activation(identity, Tracker.param(x), b)
    @test z isa Tracker.TrackedArray
end

@testset "Bias Activation: Zero-sized Arrays" begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        x = aType(rand(Float32, 4, 3, 2, 0))
        b = aType(rand(Float32, 2))
        @test size(bias_activation(identity, x, b)) == (4, 3, 2, 0)
        @test size(bias_activation!!(identity, x, b)) == (4, 3, 2, 0)

        x = aType(rand(Float32, 2, 0))
        b = aType(rand(Float32, 2))
        @test size(bias_activation(relu, x, b)) == (2, 0)
        @test size(bias_activation!!(relu, x, b)) == (2, 0)
    end
end

@testitem "Bias Activation" tags=[:other_ops] setup=[SharedTestSetup] begin
    rng = StableRNG(1234)

    bias_act_loss1(act, x, b) = sum(abs2, act.(x .+ LuxLib.Impl.reshape_bias(x, b)))
    bias_act_loss2(act, x, b) = sum(abs2, bias_activation(act, x, b))
    bias_act_loss3(act, x, b) = sum(abs2, bias_activation!!(act, copy(x), b))

    struct __Fix1{F, A}
        f::F
        act::A
    end
    (f::__Fix1)(x, b) = f.f(f.act, x, b)

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$act, $T, $sz" for act in [
                identity, relu, sigmoid, sigmoid_fast, softplus,
                logsigmoid, gelu, swish, lisht, tanh, tanh_fast],
            T in [Float16, Float32, Float64],
            sz in [(2, 2, 3, 4), (4, 5)]

            x = rand(rng, T, sz) |> aType
            b = rand(rng, T, sz[end - 1]) |> aType

            y1 = bias_act_loss1(act, x, b)
            y2 = bias_act_loss2(act, x, b)
            y3 = bias_act_loss3(act, x, b)

            fp16 = T == Float16
            atol = fp16 ? 1.0f-2 : 1.0f-3
            rtol = fp16 ? 1.0f-2 : 1.0f-3

            @test y1≈y2 atol=atol rtol=rtol
            @test y1≈y3 atol=atol rtol=rtol
            @test eltype(y1) == T
            @test eltype(y2) == T
            @test eltype(y3) == T

            @test @inferred(bias_act_loss1(act, x, b)) isa Any
            @test @inferred(bias_act_loss2(act, x, b)) isa Any
            @test @inferred(bias_act_loss3(act, x, b)) isa Any

            @jet bias_act_loss2(act, x, b)
            @jet bias_act_loss3(act, x, b)

            @test @inferred(Zygote.gradient(bias_act_loss2, act, x, b)) isa Any
            @test @inferred(Zygote.gradient(bias_act_loss3, act, x, b)) isa Any

            test_gradients(__Fix1(bias_act_loss1, act), x, b; atol, rtol,
                soft_fail=fp16 ? [AutoFiniteDiff()] : [])
            test_gradients(__Fix1(bias_act_loss2, act), x, b; atol, rtol,
                soft_fail=fp16 ? [AutoFiniteDiff()] : [])
            test_gradients(__Fix1(bias_act_loss3, act), x, b; atol, rtol,
                soft_fail=fp16 ? [AutoFiniteDiff()] : [])

            ∂x1, ∂b1 = Zygote.gradient(__Fix1(bias_act_loss1, act), x, b)
            ∂x2, ∂b2 = Zygote.gradient(__Fix1(bias_act_loss2, act), x, b)
            ∂x3, ∂b3 = Zygote.gradient(__Fix1(bias_act_loss3, act), x, b)

            @test ∂x1≈∂x2 atol=atol rtol=rtol
            @test ∂x1≈∂x3 atol=atol rtol=rtol
            @test ∂b1≈∂b2 atol=atol rtol=rtol
            @test ∂b1≈∂b3 atol=atol rtol=rtol
        end
    end
end

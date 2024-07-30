@testitem "Activation Functions" tags=[:other_ops] setup=[SharedTestSetup] begin
    rng = StableRNG(1234)

    apply_act(f::F, x) where {F} = sum(abs2, f.(x))
    apply_act_fast(f::F, x) where {F} = sum(abs2, fast_activation!!(f, copy(x)))

    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$f: $T" for f in [identity, relu, sigmoid, sigmoid_fast, softplus,
                logsigmoid, gelu, swish, lisht, tanh, tanh_fast],
            T in [Float16, Float32, Float64]

            x = rand(rng, T, 4, 3) |> aType

            y1 = apply_act(f, x)
            y2 = apply_act_fast(f, x)

            fp16 = T == Float16
            atol = fp16 ? 1.0f-1 : 1.0f-3
            rtol = fp16 ? 1.0f-1 : 1.0f-3

            @test y1≈y2 atol=atol rtol=rtol
            @test eltype(y1) == T

            @test @inferred(apply_act(f, x)) isa Any
            @test @inferred(apply_act_fast(f, x)) isa Any
            @jet apply_act_fast(f, x)
            @test @inferred(Zygote.gradient(apply_act, f, x)) isa Any

            test_gradients(Base.Fix1(apply_act, f), x; atol, rtol)
            test_gradients(Base.Fix1(apply_act_fast, f), x; atol, rtol)

            ∂x1 = Zygote.gradient(apply_act, f, x)[2]
            ∂x2 = Zygote.gradient(apply_act_fast, f, x)[2]

            @test ∂x1≈∂x2 atol=atol rtol=rtol
        end
    end
end

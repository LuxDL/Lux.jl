@testitem "Activation Functions" tags=[:other_ops] setup=[SharedTestSetup] begin
    rng = StableRNG(1234)

    apply_act(f::F, x) where {F} = sum(abs2, f.(x))
    apply_act_fast(f::F, x) where {F} = sum(abs2, fast_activation!!(f, copy(x)))
    apply_act_fast2(f::F, x) where {F} = sum(abs2, fast_activation(f, x))

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "$f: $T" for f in [identity, relu, sigmoid, sigmoid_fast, softplus,
                logsigmoid, gelu, swish, lisht, tanh, tanh_fast],
            T in [Float16, Float32, Float64]

            !fp64 && T == Float64 && continue

            x = rand(rng, T, 4, 3) |> aType

            y1 = apply_act(f, x)
            y2 = apply_act_fast(f, x)
            y3 = apply_act_fast2(f, x)

            fp16 = T == Float16
            atol = fp16 ? 1.0f-1 : 1.0f-3
            rtol = fp16 ? 1.0f-1 : 1.0f-3

            @test y1≈y2 atol=atol rtol=rtol
            @test y1≈y3 atol=atol rtol=rtol
            @test eltype(y1) == T
            @test eltype(y2) == T
            @test eltype(y3) == T

            @test @inferred(apply_act(f, x)) isa Any
            @test @inferred(apply_act_fast(f, x)) isa Any
            @test @inferred(apply_act_fast2(f, x)) isa Any

            @jet apply_act_fast(f, x)
            @jet apply_act_fast2(f, x)

            @test @inferred(Zygote.gradient(apply_act, f, x)) isa Any
            if f !== lisht
                @test @inferred(Zygote.gradient(apply_act_fast, f, x)) isa Any
            end
            @test @inferred(Zygote.gradient(apply_act_fast2, f, x)) isa Any

            @test_gradients(Base.Fix1(apply_act, f), x; atol, rtol)
            @test_gradients(Base.Fix1(apply_act_fast, f), x; atol, rtol)
            @test_gradients(Base.Fix1(apply_act_fast2, f), x; atol, rtol)

            ∂x1 = Zygote.gradient(apply_act, f, x)[2]
            ∂x2 = Zygote.gradient(apply_act_fast, f, x)[2]
            ∂x3 = Zygote.gradient(apply_act_fast2, f, x)[2]

            @test ∂x1≈∂x2 atol=atol rtol=rtol
            @test ∂x1≈∂x3 atol=atol rtol=rtol
        end
    end
end

@testitem "Fused Dense Bias Activation" tags=[:nworkers, :common_ops] setup=[SharedTestSetup] begin
    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, on_gpu) in MODES
        # These are not all possible combinations but rather a representative set to keep
        # CI timings under check
        @testset "$(Tw) x $(Tx)" for (Tw, Tx) in [
            (Float16, Float16), (Float32, Float16), (Float32, Float32),
            (Float32, Float64), (Float64, Float64)]
            for M in (4, 8),
                N in (4, 8),
                hasbias in (true, false),
                activation in (
                    identity, tanh, tanh_fast, sigmoid, sigmoid_fast, relu, gelu, x -> x^3)

                bias = hasbias ? __generate_fixed_array(Tw, M) |> aType : nothing
                w = __generate_fixed_array(Tw, M, N) |> aType
                x = __generate_fixed_array(Tx, N, 3) |> aType

                y = fused_dense_bias_activation(activation, w, x, bias)
                y_generic = LuxLib.__generic_dense_bias_activation(activation, w, x, bias)

                @test y ≈ y_generic
                @test eltype(y) == promote_type(Tw, Tx)

                @inferred fused_dense_bias_activation(activation, w, x, bias)
                @jet fused_dense_bias_activation(activation, w, x, bias)

                __f = (σ, w, x, b) -> sum(abs2, fused_dense_bias_activation(σ, w, x, b))
                fp16 = Tx == Float16 || Tw == Float16
                atol = fp16 ? 1.0f-1 : 1.0f-3
                rtol = fp16 ? 1.0f-1 : 1.0f-3
                # FiniteDiffencing doesn't work great for MP because of how LuxTestUtils is
                # implemented.
                @eval @test_gradients $__f $activation $w $x $bias gpu_testing=$on_gpu soft_fail=$fp16 atol=$atol rtol=$rtol skip_finite_differences=$(Tx !=
                                                                                                                                                       Tw)
            end
        end
    end
end

@testitem "Scaled Dot Product Attention" tags = [:misc] setup = [SharedTestSetup] begin
    using LuxLib, Reactant, NNlib, Random, MLDataDevices, Zygote, Enzyme, Statistics

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "Different Batch Sizes" begin
            n, lenq, lenkv = 15, 3, 4

            @testset for batch_size in [1, 2, (2, 1, 3)], nheads in [1, 3, 5]
                q =
                    Reactant.TestUtils.construct_test_array(
                        Float32, n ÷ nheads, nheads, lenq, batch_size...
                    ) |> aType
                k =
                    Reactant.TestUtils.construct_test_array(
                        Float32, n ÷ nheads, nheads, lenkv, batch_size...
                    ) |> aType
                v =
                    Reactant.TestUtils.construct_test_array(
                        Float32, n ÷ nheads, nheads, lenkv, batch_size...
                    ) |> aType
                y1, α1 = scaled_dot_product_attention(q, k, v) |> cpu_device()

                @test y1 isa Array{Float32,length(batch_size) + 3}
                @test size(y1) == (n ÷ nheads, nheads, lenq, batch_size...)
                @test size(α1) == (lenkv, lenq, nheads, batch_size...)
                @test sum(α1; dims=1) ≈ ones(1, lenq, nheads, batch_size...)

                q_ra = Reactant.to_rarray(q)
                k_ra = Reactant.to_rarray(k)
                v_ra = Reactant.to_rarray(v)
                y1_ra, α1_ra = @jit scaled_dot_product_attention(q_ra, k_ra, v_ra)

                @test y1_ra ≈ y1 atol = 1.0f-3 rtol = 1.0f-3
                @test α1_ra ≈ α1 atol = 1.0f-3 rtol = 1.0f-3

                @testset "Gradient Check" begin
                    ∂q_fd, ∂k_fd, ∂v_fd = @jit Reactant.TestUtils.finite_difference_gradient(
                        sum ∘ first ∘ scaled_dot_product_attention,
                        Float64.(q_ra),
                        Float64.(k_ra),
                        Float64.(v_ra),
                    )
                    ∂q_reactant, ∂k_reactant, ∂v_reactant = @jit Enzyme.gradient(
                        Reverse,
                        Const(sum ∘ first ∘ scaled_dot_product_attention),
                        q_ra,
                        k_ra,
                        v_ra,
                    )

                    @test ∂q_fd ≈ ∂q_reactant atol = 1e-3 rtol = 1e-3
                    @test ∂k_fd ≈ ∂k_reactant atol = 1e-3 rtol = 1e-3
                    @test ∂v_fd ≈ ∂v_reactant atol = 1e-3 rtol = 1e-3
                end

                function sdpa(q, k, v)
                    return scaled_dot_product_attention(q, k, v; token_dim=1, head_dim=3)
                end

                q2 = permutedims(q, (3, 2, 1, 4:(3 + length(batch_size))...)) |> aType
                k2 = permutedims(k, (3, 2, 1, 4:(3 + length(batch_size))...)) |> aType
                v2 = permutedims(v, (3, 2, 1, 4:(3 + length(batch_size))...)) |> aType
                y2, α2 = sdpa(q2, k2, v2) |> cpu_device()

                @test y2 isa Array{Float32,length(batch_size) + 3}
                @test size(y2) == (n ÷ nheads, nheads, lenq, batch_size...)
                @test size(α2) == (lenkv, lenq, nheads, batch_size...)
                @test sum(α2; dims=1) ≈ ones(1, lenq, nheads, batch_size...)

                q2_ra = Reactant.to_rarray(q2)
                k2_ra = Reactant.to_rarray(k2)
                v2_ra = Reactant.to_rarray(v2)
                y2_ra, α2_ra = @jit sdpa(q2_ra, k2_ra, v2_ra)

                @test y2_ra ≈ y2 atol = 1.0f-3 rtol = 1.0f-3
                @test α2_ra ≈ α2 atol = 1.0f-3 rtol = 1.0f-3

                @testset "Gradient Check" begin
                    ∂q_fd, ∂k_fd, ∂v_fd = @jit Reactant.TestUtils.finite_difference_gradient(
                        sum ∘ first ∘ sdpa,
                        Float64.(q2_ra),
                        Float64.(k2_ra),
                        Float64.(v2_ra),
                    )
                    ∂q2_reactant, ∂k2_reactant, ∂v2_reactant = @jit Enzyme.gradient(
                        Reverse, Const(sum ∘ first ∘ sdpa), q2_ra, k2_ra, v2_ra
                    )

                    @test ∂q_fd ≈ ∂q2_reactant atol = 1.0f-3 rtol = 1.0f-3
                    @test ∂k_fd ≈ ∂k2_reactant atol = 1.0f-3 rtol = 1.0f-3
                    @test ∂v_fd ≈ ∂v2_reactant atol = 1.0f-3 rtol = 1.0f-3
                end

                @test y1 ≈ y2 atol = 1.0f-3 rtol = 1.0f-3
                @test α1 ≈ α2 atol = 1.0f-3 rtol = 1.0f-3
                @test y1_ra ≈ y2_ra atol = 1.0f-3 rtol = 1.0f-3
                @test α1_ra ≈ α2_ra atol = 1.0f-3 rtol = 1.0f-3
            end

            @testset "Specific Results Check" begin
                q = k = v = aType(reshape([1:12;], 2, 2, 3, 1) ./ 12)
                y, α = scaled_dot_product_attention(q, k, v) |> cpu_device()

                q_ra = Reactant.to_rarray(q)
                k_ra = Reactant.to_rarray(k)
                v_ra = Reactant.to_rarray(v)
                y_ra, α_ra = @jit scaled_dot_product_attention(q_ra, k_ra, v_ra)

                ytrue = reshape(
                    [
                        0.429754,
                        0.513087,
                        0.613791,
                        0.697125,
                        0.46431,
                        0.547644,
                        0.647876,
                        0.73121,
                        0.49773,
                        0.581064,
                        0.680455,
                        0.763788,
                    ],
                    2,
                    2,
                    3,
                    1,
                )
                αtrue = reshape(
                    [
                        0.313896,
                        0.332948,
                        0.353157,
                        0.264431,
                        0.328206,
                        0.407362,
                        0.219215,
                        0.31838,
                        0.462405,
                        0.288691,
                        0.331243,
                        0.380066,
                        0.241239,
                        0.323893,
                        0.434868,
                        0.198438,
                        0.311761,
                        0.489801,
                    ],
                    3,
                    3,
                    2,
                    1,
                )

                @test y ≈ ytrue atol = 1e-5 rtol = 1e-5
                @test α ≈ αtrue atol = 1e-5 rtol = 1e-5
                @test y_ra ≈ ytrue atol = 1e-5 rtol = 1e-5
                @test α_ra ≈ αtrue atol = 1e-5 rtol = 1e-5
            end

            @testset "Masking" begin
                sdpa_with_mask(q, k, v, mask) = scaled_dot_product_attention(q, k, v; mask)

                q = aType(rand(Float32, 4, 2, 3, 1))
                k = aType(rand(Float32, 4, 2, 5, 1))
                v = aType(rand(Float32, 4, 2, 5, 1))
                mask = aType(rand(Bool, (5, 3)))

                y, α = sdpa_with_mask(q, k, v, mask) |> cpu_device()

                @test all((α[:, :, 1, 1] .> 0) .== Array(mask))
                @test all((α[:, :, 2, 1] .> 0) .== Array(mask))

                q_ra = Reactant.to_rarray(q)
                k_ra = Reactant.to_rarray(k)
                v_ra = Reactant.to_rarray(v)
                mask_ra = Reactant.to_rarray(mask)

                y_ra, α_ra = @jit sdpa_with_mask(q_ra, k_ra, v_ra, mask_ra)

                @test all((Array(α_ra)[:, :, 1, 1] .> 0) .== Array(mask))
                @test all((Array(α_ra)[:, :, 2, 1] .> 0) .== Array(mask))

                @testset "causal" begin
                    mask = LuxLib.Impl.make_causal_mask(q, size(k, 3), size(q, 3))
                    mask_ra = Reactant.to_rarray(mask)

                    y, α = sdpa_with_mask(q, k, v, mask) |> cpu_device()
                    y_ra, α_ra = @jit sdpa_with_mask(q_ra, k_ra, v_ra, mask_ra)

                    @test y ≈ y_ra atol = 1e-5 rtol = 1e-5
                    @test α ≈ α_ra atol = 1e-5 rtol = 1e-5

                    y2, α2 =
                        scaled_dot_product_attention(q, k, v; is_causal=true) |>
                        cpu_device()
                    y_ra2, α_ra2 = @jit scaled_dot_product_attention(
                        q_ra, k_ra, v_ra; is_causal=true
                    )

                    @test y2 ≈ y_ra2 atol = 1e-5 rtol = 1e-5
                    @test α2 ≈ α_ra2 atol = 1e-5 rtol = 1e-5

                    @test y ≈ y2 atol = 1e-5 rtol = 1e-5
                    @test α ≈ α2 atol = 1e-5 rtol = 1e-5
                    @test y_ra ≈ y_ra2 atol = 1e-5 rtol = 1e-5
                    @test α_ra ≈ α_ra2 atol = 1e-5 rtol = 1e-5

                    @test_throws AssertionError scaled_dot_product_attention(
                        q, k, v; mask, is_causal=true
                    )
                end
            end

            @testset "Bias" begin
                q = aType(rand(Float32, 2, 2, 5, 1))
                k = v = aType(rand(Float32, 2, 2, 3, 1))
                bias = aType(randn(Float32, 3, 5))

                sdpa_with_bias(q, k, v, bias) = scaled_dot_product_attention(q, k, v; bias)

                y, α = sdpa_with_bias(q, k, v, bias) |> cpu_device()

                @test size(α) == (3, 5, 2, 1)
                @test size(y) == (2, 2, 5, 1)

                q_ra = Reactant.to_rarray(q)
                k_ra = Reactant.to_rarray(k)
                v_ra = Reactant.to_rarray(v)
                bias_ra = Reactant.to_rarray(bias)

                y_ra, α_ra = @jit sdpa_with_bias(q_ra, k_ra, v_ra, bias_ra)

                @test y ≈ y_ra atol = 1e-5 rtol = 1e-5
                @test α ≈ α_ra atol = 1e-5 rtol = 1e-5
            end

            @testset "Dropout" begin
                q = k = v = aType(rand(Float32, 5, 2, 10, 10))
                fdrop(x, p) = (rand!(similar(x)) .> p) .* x ./ (1 - p)

                y, α =
                    scaled_dot_product_attention(q, k, v; fdrop=Base.Fix2(fdrop, 0.5f0)) |>
                    cpu_device()
                @test 0.6 > mean(>(0), α) > 0.4
            end

            @testset "Grouped KV" begin
                q = aType(rand(Float32, 4, 8, 5, 3))
                k = aType(rand(Float32, 4, 2, 7, 3))
                v = aType(rand(Float32, 11, 2, 7, 3))

                y, α = scaled_dot_product_attention(q, k, v) |> cpu_device()

                @test size(y) == (11, 8, 5, 3)
                @test size(α) == (7, 5, 8, 3)

                q_ra = Reactant.to_rarray(q)
                k_ra = Reactant.to_rarray(k)
                v_ra = Reactant.to_rarray(v)
                y_ra, α_ra = @jit scaled_dot_product_attention(q_ra, k_ra, v_ra)

                @test y ≈ y_ra atol = 1e-3 rtol = 1e-3
                @test α ≈ α_ra atol = 1e-3 rtol = 1e-3
            end
        end
    end
end

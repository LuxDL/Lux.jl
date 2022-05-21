using JET, Lux, NNlib, Random, Statistics, Zygote

include("../utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

# @testset "BatchNorm" begin
#     let m = BatchNorm(2), x = [
#             1.0f0 3.0f0 5.0f0
#             2.0f0 4.0f0 6.0f0
#         ]
#         ps, st = Lux.setup(rng, m)

#         @test Lux.parameterlength(m) == 2 * 2

#         @test ps.bias == [0, 0]  # init_bias(2)
#         @test ps.scale == [1, 1]  # init_scale(2)

#         y, st_ = pullback(m, x, ps, st)[1]
#         @test isapprox(y, [-1.22474 0 1.22474; -1.22474 0 1.22474], atol=1.0e-5)
#         # julia> x
#         #  2×3 Array{Float64,2}:
#         #  1.0  3.0  5.0
#         #  2.0  4.0  6.0
#         #
#         # mean of batch will be
#         #  (1. + 3. + 5.) / 3 = 3
#         #  (2. + 4. + 6.) / 3 = 4
#         #
#         # ∴ update rule with momentum:
#         #  .1 * 3 + 0 = .3
#         #  .1 * 4 + 0 = .4
#         @test st_.running_mean ≈ reshape([0.3, 0.4], 2, 1)

#         # julia> .1 .* var(x, dims = 2, corrected=false) .* (3 / 2).+ .9 .* [1., 1.]
#         # 2×1 Array{Float64,2}:
#         #  1.3
#         #  1.3
#         @test st_.running_var ≈ 0.1 .* var(x; dims=2, corrected=false) .* (3 / 2) .+ 0.9 .* [1.0, 1.0]

#         st_ = Lux.testmode(st_)
#         x′ = m(x, ps, st_)[1]
#         @test isapprox(x′[1], (1 .- 0.3) / sqrt(1.3), atol=1.0e-5)

#         @inferred m(x, ps, st)

#         @test_call m(x, ps, st)
#         @test_opt target_modules = (Lux,) m(x, ps, st)

#         test_gradient_correctness_fdm((x, ps) -> sum(first(m(x, ps, st))), x, ps; atol=1.0f-3, rtol=1.0f-3)
#     end

#     let m = BatchNorm(2; track_stats=false), x = [1.0f0 3.0f0 5.0f0; 2.0f0 4.0f0 6.0f0]
#         ps, st = Lux.setup(rng, m)
#         @inferred m(x, ps, st)
#         @test_call m(x, ps, st)
#         @test_opt target_modules = (Lux,) m(x, ps, st)

#         test_gradient_correctness_fdm((x, ps) -> sum(first(m(x, ps, st))), x, ps; atol=1.0f-3, rtol=1.0f-3)
#     end

#     # with activation function
#     let m = BatchNorm(2, sigmoid), x = [
#             1.0f0 3.0f0 5.0f0
#             2.0f0 4.0f0 6.0f0
#         ]
#         ps, st = Lux.setup(rng, m)
#         st = Lux.testmode(st)
#         y, st_ = m(x, ps, st)
#         @test isapprox(y, sigmoid.((x .- st_.running_mean) ./ sqrt.(st_.running_var .+ m.epsilon)), atol=1.0e-7)
#         @inferred m(x, ps, st)
#         @test_call m(x, ps, st)
#         @test_opt target_modules = (Lux,) m(x, ps, st)

#         test_gradient_correctness_fdm((x, ps) -> sum(first(m(x, ps, st))), x, ps; atol=1.0f-3, rtol=1.0f-3)
#     end

#     let m = BatchNorm(2), x = reshape(Float32.(1:6), 3, 2, 1)
#         ps, st = Lux.setup(rng, m)
#         st = Lux.trainmode(st)
#         @test_throws AssertionError m(x, ps, st)[1]
#     end

#     let m = BatchNorm(32), x = randn(Float32, 416, 416, 32, 1)
#         ps, st = Lux.setup(rng, m)
#         st = Lux.testmode(st)
#         m(x, ps, st)
#         @test (@allocated m(x, ps, st)) < 100_000_000
#         @inferred m(x, ps, st)
#         @test_call m(x, ps, st)
#         @test_opt target_modules = (Lux,) m(x, ps, st)
#     end
# end

@testset "GroupNorm" begin
    # begin tests
    squeeze(x) = dropdims(x; dims=tuple(findall(size(x) .== 1)...)) # To remove all singular dimensions

    let m = GroupNorm(4, 2; track_stats=true), sizes = (3, 4, 2), x = reshape(collect(1:prod(sizes)), sizes)
        @test Lux.parameterlength(m) == 2 * 4
        x = Float32.(x)
        ps, st = Lux.setup(rng, m)
        @test ps.bias == [0, 0, 0, 0]   # init_bias(32)
        @test ps.scale == [1, 1, 1, 1]  # init_scale(32)

        y, st_ = pullback(m, x, ps, st)[1]

        # julia> x
        # [:, :, 1]  =
        # 1.0  4.0  7.0  10.0
        # 2.0  5.0  8.0  11.0
        # 3.0  6.0  9.0  12.0
        #
        # [:, :, 2] =
        # 13.0  16.0  19.0  22.0
        # 14.0  17.0  20.0  23.0
        # 15.0  18.0  21.0  24.0
        #
        # mean will be
        # (1. + 2. + 3. + 4. + 5. + 6.) / 6 = 3.5
        # (7. + 8. + 9. + 10. + 11. + 12.) / 6 = 9.5
        #
        # (13. + 14. + 15. + 16. + 17. + 18.) / 6 = 15.5
        # (19. + 20. + 21. + 22. + 23. + 24.) / 6 = 21.5
        #
        # mean =
        # 3.5   15.5
        # 9.5   21.5
        #
        # ∴ update rule with momentum:
        # (1. - .1) * 0 + .1 * (3.5 + 15.5) / 2 = 0.95
        # (1. - .1) * 0 + .1 * (9.5 + 21.5) / 2 = 1.55
        @test st_.running_mean ≈ [0.95, 1.55]
        n = prod(size(x)) ÷ m.groups ÷ size(x)[end]
        corr = n / (n - 1)
        z = reshape(x, 3, 2, 2, 2)
        variance = var(z; dims=(1, 2), corrected=false)
        @test st_.running_var ≈ 0.1 * corr * vec(mean(variance; dims=4)) .+ 0.9 * 1

        st__ = Lux.testmode(st_)
        y, st__ = m(x, ps, st__)
        out = (z .- reshape(st_.running_mean, 1, 1, 2, 1)) ./ sqrt.(reshape(st_.running_var, 1, 1, 2, 1) .+ 1.0f-5)
        @test y ≈ reshape(out, size(x)) atol = 1.0e-5

        @inferred m(x, ps, st)
        @test_call m(x, ps, st)
        @test_opt target_modules = (Lux,) m(x, ps, st)
        test_gradient_correctness_fdm(ps -> sum(first(m(x, ps, st))), ps; atol=1.0f-3, rtol=1.0f-3)
    end

    # with activation function
    let m = GroupNorm(4, 2, sigmoid; track_stats=true), sizes = (3, 4, 2), x = reshape(collect(1:prod(sizes)), sizes)
        # x = Float32.(x)
    #     μ_affine_shape = ones(Int, length(sizes) + 1)
    #     μ_affine_shape[end - 1] = 2 # Number of groups

    #     affine_shape = ones(Int, length(sizes) + 1)
    #     affine_shape[end - 2] = 2 # Channels per group
    #     affine_shape[end - 1] = 2 # Number of groups
    #     affine_shape[1] = sizes[1]
    #     affine_shape[end] = sizes[end]

    #     og_shape = size(x)

    #     y = m(x)
    #     x_ = reshape(x, affine_shape...)
    #     out = reshape(
    #         sigmoid.((x_ .- reshape(m.μ, μ_affine_shape...)) ./ sqrt.(reshape(m.σ², μ_affine_shape...) .+ m.ϵ)),
    #         og_shape,
    #     )
    #     @test y ≈ out atol = 1e-7
    end

    # let m = trainmode!(GroupNorm(2, 2; track_stats=true)),
    #     sizes = (2, 4, 1, 2, 3),
    #     x = Float32.(reshape(collect(1:prod(sizes)), sizes))

    #     y = reshape(permutedims(x, [3, 1, 2, 4, 5]), :, 2, 3)
    #     y = reshape(m(y), sizes...)
    #     @test m(x) == y
    # end

    # # check that μ, σ², and the output are the correct size for higher rank tensors
    # let m = GroupNorm(4, 2; track_stats=true),
    #     sizes = (5, 5, 3, 4, 4, 6),
    #     x = Float32.(reshape(collect(1:prod(sizes)), sizes))

    #     y = evalwgrad(m, x)
    #     @test size(m.μ) == (m.G,)
    #     @test size(m.σ²) == (m.G,)
    #     @test size(y) == sizes
    # end

    # # show that group norm is the same as instance norm when the group size is the same as the number of channels
    # let IN = trainmode!(InstanceNorm(4; affine=true)),
    #     GN = trainmode!(GroupNorm(4, 4)),
    #     sizes = (2, 2, 3, 4, 5),
    #     x = Float32.(reshape(collect(1:prod(sizes)), sizes))

    #     @test IN(x) ≈ GN(x)
    # end

    # # show that group norm is the same as batch norm for a group of size 1 and batch of size 1
    # let BN = trainmode!(BatchNorm(4)),
    #     GN = trainmode!(GroupNorm(4, 4)),
    #     sizes = (2, 2, 3, 4, 1),
    #     x = Float32.(reshape(collect(1:prod(sizes)), sizes))

    #     @test BN(x) ≈ GN(x)
    # end
end

@testset "BatchNorm" begin
    let m = BatchNorm(2), x = [
            1.0f0 3.0f0 5.0f0
            2.0f0 4.0f0 6.0f0
        ]

        ps, st = setup(MersenneTwister(0), m)

        @test parameterlength(m) == 2 * 2

        @test ps.β == [0, 0]  # initβ(2)
        @test ps.γ == [1, 1]  # initγ(2)

        y, st_ = pullback(m, x, ps, st)[1]
        @test isapprox(y, [-1.22474 0 1.22474; -1.22474 0 1.22474], atol=1.0e-5)
        # julia> x
        #  2×3 Array{Float64,2}:
        #  1.0  3.0  5.0
        #  2.0  4.0  6.0
        #
        # μ of batch will be
        #  (1. + 3. + 5.) / 3 = 3
        #  (2. + 4. + 6.) / 3 = 4
        #
        # ∴ update rule with momentum:
        #  .1 * 3 + 0 = .3
        #  .1 * 4 + 0 = .4
        @test st_.μ ≈ reshape([0.3, 0.4], 2, 1)

        # julia> .1 .* var(x, dims = 2, corrected=false) .* (3 / 2).+ .9 .* [1., 1.]
        # 2×1 Array{Float64,2}:
        #  1.3
        #  1.3
        @test st_.σ² ≈ 0.1 .* var(x; dims=2, corrected=false) .* (3 / 2) .+ 0.9 .* [1.0, 1.0]

        st_ = testmode(st_)
        x′ = m(x, ps, st_)[1]
        @test isapprox(x′[1], (1 .- 0.3) / sqrt(1.3), atol=1.0e-5)

        @inferred m(x, ps, st)
    end

    let m = BatchNorm(2; track_stats=false), x = [1.0f0 3.0f0 5.0f0; 2.0f0 4.0f0 6.0f0]
        ps, st = setup(MersenneTwister(0), m)
        @inferred m(x, ps, st)
    end

    # with activation function
    let m = BatchNorm(2, sigmoid), x = [
            1.0f0 3.0f0 5.0f0
            2.0f0 4.0f0 6.0f0
        ]
        ps, st = setup(MersenneTwister(0), m)
        st = testmode(st)
        y, st_ = m(x, ps, st)
        @test isapprox(y, sigmoid.((x .- st_.μ) ./ sqrt.(st_.σ² .+ m.ϵ)), atol=1.0e-7)
        @inferred m(x, ps, st)
    end

    let m = BatchNorm(2), x = reshape(Float32.(1:6), 3, 2, 1)
        ps, st = setup(MersenneTwister(0), m)
        st = trainmode(st)
        @test_throws AssertionError m(x, ps, st)[1]
    end

    let m = BatchNorm(32), x = randn(Float32, 416, 416, 32, 1)
        ps, st = setup(MersenneTwister(0), m)
        st = testmode(st)
        m(x, ps, st)
        @test (@allocated m(x, ps, st)) < 100_000_000
        @inferred m(x, ps, st)
    end
end

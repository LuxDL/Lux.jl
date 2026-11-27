using Enzyme

include("../shared_testsetup.jl")

@testset "BatchNorm" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        m = BatchNorm(2)
        x = aType([
            1.0f0 3.0f0 5.0f0
            2.0f0 4.0f0 6.0f0
        ])
        display(m)
        ps, st = dev(Lux.setup(rng, m))

        @test Lux.parameterlength(m) == Lux.parameterlength(ps)
        @test Lux.statelength(m) == Lux.statelength(st)

        @test ps.bias == aType([0, 0])  # init_bias(2)
        @test ps.scale == aType([1, 1])  # init_scale(2)

        y, st_ = pullback(m, x, ps, st)[1]
        st_ = CPUDevice()(st_)
        @test Array(y) ≈ [-1.22474 0 1.22474; -1.22474 0 1.22474] atol = 1.0e-5
        # julia> x
        #  2×3 Array{Float64,2}:
        #  1.0  3.0  5.0
        #  2.0  4.0  6.0

        # mean of batch will be
        #  (1. + 3. + 5.) / 3 = 3
        #  (2. + 4. + 6.) / 3 = 4

        # ∴ update rule with momentum:
        #  .1 * 3 + 0 = .3
        #  .1 * 4 + 0 = .4
        @test st_.running_mean ≈ reshape([0.3, 0.4], 2, 1)

        # julia> .1 .* var(x, dims = 2, corrected=false) .* (3 / 2).+ .9 .* [1., 1.]
        # 2×1 Array{Float64,2}:
        #  1.3
        #  1.3
        @test st_.running_var ≈
            0.1 .* var(Array(x); dims=2, corrected=false) .* (3 / 2) .+ 0.9 .* [1.0, 1.0]

        st_ = dev(Lux.testmode(st_))
        x_ = CPUDevice()(m(x, ps, st_)[1])
        @test x_[1] ≈ (1 .- 0.3) / sqrt(1.3) atol = 1.0e-5

        # chunking will cause incorrect gradients
        skip_backends = [AutoFiniteDiff(), AutoForwardDiff()]

        @jet m(x, ps, st)
        @test_gradients(sumabs2first, m, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends)

        @testset for affine in (true, false)
            m = BatchNorm(2; affine, track_stats=false)
            x = aType([1.0f0 3.0f0 5.0f0; 2.0f0 4.0f0 6.0f0])
            display(m)
            ps, st = dev(Lux.setup(rng, m))

            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(
                sumabs2first, m, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )

            # with activation function
            m = BatchNorm(2, sigmoid; affine)
            x = aType([
                1.0f0 3.0f0 5.0f0
                2.0f0 4.0f0 6.0f0
            ])
            display(m)
            ps, st = dev(Lux.setup(rng, m))

            y, st_ = m(x, ps, Lux.testmode(st))
            @test y ≈
                sigmoid.((x .- st_.running_mean) ./ sqrt.(st_.running_var .+ m.epsilon))
            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(
                sumabs2first, m, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )

            m = BatchNorm(32; affine)
            x = aType(randn(Float32, 32, 32, 32, 1))
            display(m)
            ps, st = dev(Lux.setup(rng, m))
            m(x, ps, Lux.testmode(st))
            @jet m(x, ps, Lux.testmode(st))
        end
    end
end

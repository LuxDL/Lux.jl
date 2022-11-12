using Lux, NNlib, Random, Statistics, Zygote

include("../test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "BatchNorm" begin
    m = BatchNorm(2)
    x = [1.0f0 3.0f0 5.0f0
         2.0f0 4.0f0 6.0f0]
    display(m)
    ps, st = Lux.setup(rng, m)

    @test Lux.parameterlength(m) == Lux.parameterlength(ps)
    @test Lux.statelength(m) == Lux.statelength(st)

    @test ps.bias == [0, 0]  # init_bias(2)
    @test ps.scale == [1, 1]  # init_scale(2)

    y, st_ = pullback(m, x, ps, st)[1]
    @test isapprox(y, [-1.22474 0 1.22474; -1.22474 0 1.22474], atol=1.0e-5)
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
          0.1 .* var(x; dims=2, corrected=false) .* (3 / 2) .+ 0.9 .* [1.0, 1.0]

    st_ = Lux.testmode(st_)
    x_ = m(x, ps, st_)[1]
    @test isapprox(x_[1], (1 .- 0.3) / sqrt(1.3), atol=1.0e-5)

    @inferred first(m(x, ps, st))

    run_JET_tests(m, x, ps, st)

    test_gradient_correctness_fdm((x, ps) -> sum(first(m(x, ps, st))), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    for affine in (true, false)
        m = BatchNorm(2; affine, track_stats=false)
        x = [1.0f0 3.0f0 5.0f0; 2.0f0 4.0f0 6.0f0]
        display(m)
        ps, st = Lux.setup(rng, m)
        @inferred first(m(x, ps, st))
        run_JET_tests(m, x, ps, st)

        if affine
            test_gradient_correctness_fdm((x, ps) -> sum(first(m(x, ps, st))), x, ps;
                                          atol=1.0f-3, rtol=1.0f-3)
        else
            test_gradient_correctness_fdm(x -> sum(first(m(x, ps, st))), x; atol=1.0f-3,
                                          rtol=1.0f-3)
        end

        # with activation function
        m = BatchNorm(2, sigmoid; affine)
        x = [1.0f0 3.0f0 5.0f0
             2.0f0 4.0f0 6.0f0]
        display(m)
        ps, st = Lux.setup(rng, m)
        st = Lux.testmode(st)
        y, st_ = m(x, ps, st)
        @test isapprox(y,
                       sigmoid.((x .- st_.running_mean) ./
                                sqrt.(st_.running_var .+ m.epsilon)), atol=1.0e-7)
        @inferred first(m(x, ps, st))
        run_JET_tests(m, x, ps, st)

        if affine
            test_gradient_correctness_fdm((x, ps) -> sum(first(m(x, ps, st))), x, ps;
                                          atol=1.0f-3, rtol=1.0f-3)
        else
            test_gradient_correctness_fdm(x -> sum(first(m(x, ps, st))), x; atol=1.0f-3,
                                          rtol=1.0f-3)
        end

        m = BatchNorm(32; affine)
        x = randn(Float32, 416, 416, 32, 1)
        display(m)
        ps, st = Lux.setup(rng, m)
        st = Lux.testmode(st)
        m(x, ps, st)
        @test (@allocated m(x, ps, st)) < 100_000_000
        @inferred first(m(x, ps, st))
        run_JET_tests(m, x, ps, st)
    end

    @testset "allow fast activation" begin
        layer = BatchNorm(10, tanh)
        @test layer.activation == tanh_fast
        layer = BatchNorm(10, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end
end

@testset "GroupNorm" begin
    # begin tests
    squeeze(x) = dropdims(x; dims=tuple(findall(size(x) .== 1)...)) # To remove all singular dimensions

    m = GroupNorm(4, 2; track_stats=true)
    sizes = (3, 4, 2)
    x = reshape(collect(1:prod(sizes)), sizes)

    display(m)
    x = Float32.(x)
    ps, st = Lux.setup(rng, m)
    @test Lux.parameterlength(m) == Lux.parameterlength(ps)
    @test Lux.statelength(m) == Lux.statelength(st)
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
    out = (z .- reshape(st_.running_mean, 1, 1, 2, 1)) ./
          sqrt.(reshape(st_.running_var, 1, 1, 2, 1) .+ 1.0f-5)
    @test y≈reshape(out, size(x)) atol=1.0e-5

    @inferred first(m(x, ps, st))
    run_JET_tests(m, x, ps, st)
    test_gradient_correctness_fdm(ps -> sum(first(m(x, ps, st))), ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    for affine in (true, false)
        m = GroupNorm(2, 2; affine, track_stats=false)
        x = randn(rng, Float32, 3, 2, 1)
        display(m)
        ps, st = Lux.setup(rng, m)
        @inferred first(m(x, ps, st))
        run_JET_tests(m, x, ps, st)

        if affine
            test_gradient_correctness_fdm((x, ps) -> sum(first(m(x, ps, st))), x, ps;
                                          atol=1.0f-3, rtol=1.0f-3)
        else
            test_gradient_correctness_fdm(x -> sum(first(m(x, ps, st))), x; atol=1.0f-3,
                                          rtol=1.0f-3)
        end

        # with activation function
        m = GroupNorm(2, 2, sigmoid; affine)
        x = randn(rng, Float32, 3, 2, 1)
        display(m)
        ps, st = Lux.setup(rng, m)
        st = Lux.testmode(st)
        y, st_ = m(x, ps, st)

        @inferred first(m(x, ps, st))
        run_JET_tests(m, x, ps, st)

        if affine
            test_gradient_correctness_fdm((x, ps) -> sum(first(m(x, ps, st))), x, ps;
                                          atol=1.0f-3, rtol=1.0f-3)
        else
            test_gradient_correctness_fdm(x -> sum(first(m(x, ps, st))), x; atol=1.0f-3,
                                          rtol=1.0f-3)
        end

        m = GroupNorm(32, 16; affine)
        x = randn(rng, Float32, 416, 416, 32, 1)
        display(m)
        ps, st = Lux.setup(rng, m)
        st = Lux.testmode(st)
        m(x, ps, st)
        @test (@allocated m(x, ps, st)) < 100_000_000
        @inferred first(m(x, ps, st))
        run_JET_tests(m, x, ps, st)
    end

    @test_throws AssertionError GroupNorm(5, 2)

    @testset "allow fast activation" begin
        layer = GroupNorm(10, 2, tanh)
        @test layer.activation == tanh_fast
        layer = GroupNorm(10, 2, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end

    # Deprecated Functionality (remove in v0.5)
    @test_deprecated GroupNorm(4, 2; track_stats=true)
    @test_deprecated GroupNorm(4, 2; track_stats=false, momentum=0.3f0)
end

@testset "WeightNorm" begin
    @testset "_norm_except" begin
        z = randn(rng, Float32, 3, 3, 4, 2)

        @test size(Lux._norm(z; dims=(1, 2))) == (1, 1, 4, 2)
        @test size(Lux._norm_except(z; dims=1)) == (3, 1, 1, 1)
        @test Lux._norm_except(z; dims=2) == Lux._norm(z; dims=(1, 3, 4))
        @test size(Lux._norm_except(z; dims=(1, 2))) == (3, 3, 1, 1)
        @test Lux._norm_except(z; dims=(1, 2)) == Lux._norm(z; dims=(3, 4))

        run_JET_tests(Lux._norm_except, z)
        run_JET_tests(x -> Lux._norm_except(x; dims=(3, 4)), z)
    end

    @testset "Conv" begin
        c = Conv((3, 3), 3 => 3; init_bias=Lux.ones32)

        wn = WeightNorm(c, (:weight, :bias))
        display(wn)
        ps, st = Lux.setup(rng, wn)
        x = randn(rng, Float32, 3, 3, 3, 1)

        run_JET_tests(wn, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(first(wn(x, ps, st))), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        wn = WeightNorm(c, (:weight,))
        display(wn)
        ps, st = Lux.setup(rng, wn)
        x = randn(rng, Float32, 3, 3, 3, 1)

        run_JET_tests(wn, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(first(wn(x, ps, st))), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        wn = WeightNorm(c, (:weight, :bias), (2, 2))
        display(wn)
        ps, st = Lux.setup(rng, wn)
        x = randn(rng, Float32, 3, 3, 3, 1)

        run_JET_tests(wn, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(first(wn(x, ps, st))), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        wn = WeightNorm(c, (:weight,), (2,))
        display(wn)
        ps, st = Lux.setup(rng, wn)
        x = randn(rng, Float32, 3, 3, 3, 1)

        run_JET_tests(wn, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(first(wn(x, ps, st))), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end

    @testset "Dense" begin
        d = Dense(3 => 3; init_bias=Lux.ones32)

        wn = WeightNorm(d, (:weight, :bias))
        display(wn)
        ps, st = Lux.setup(rng, wn)
        x = randn(rng, Float32, 3, 1)

        run_JET_tests(wn, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(first(wn(x, ps, st))), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        wn = WeightNorm(d, (:weight,))
        display(wn)
        ps, st = Lux.setup(rng, wn)
        x = randn(rng, Float32, 3, 1)

        run_JET_tests(wn, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(first(wn(x, ps, st))), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        wn = WeightNorm(d, (:weight, :bias), (2, 2))
        display(wn)
        ps, st = Lux.setup(rng, wn)
        x = randn(rng, Float32, 3, 1)

        run_JET_tests(wn, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(first(wn(x, ps, st))), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        wn = WeightNorm(d, (:weight,), (2,))
        display(wn)
        ps, st = Lux.setup(rng, wn)
        x = randn(rng, Float32, 3, 1)

        run_JET_tests(wn, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(first(wn(x, ps, st))), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end

    # See https://github.com/avik-pal/Lux.jl/issues/95
    @testset "Normalizing Zero Parameters" begin
        c = Conv((3, 3), 3 => 3)

        wn = WeightNorm(c, (:weight, :bias))
        @test_throws ArgumentError Lux.setup(rng, wn)

        wn = WeightNorm(c, (:weight,))
        @test_nowarn Lux.setup(rng, wn)

        c = Conv((3, 3), 3 => 3; init_bias=Lux.ones32)

        wn = WeightNorm(c, (:weight, :bias))
        @test_nowarn Lux.setup(rng, wn)

        wn = WeightNorm(c, (:weight,))
        @test_nowarn Lux.setup(rng, wn)
    end
end

@testset "LayerNorm" begin
    x = randn(rng, Float32, 3, 3, 3, 2)

    for bshape in ((3, 3, 3), (1, 3, 1), (3, 1, 3))
        for affine in (true, false)
            ln = LayerNorm(bshape; affine)
            display(ln)
            ps, st = Lux.setup(rng, ln)

            @inferred first(ln(x, ps, st))
            y, st_ = ln(x, ps, st)

            @test isapprox(mean(y), 0; atol=1.0f-3, rtol=1.0f-3)
            @test isapprox(std(y), 1; atol=1.0f-2, rtol=1.0f-2)

            run_JET_tests(ln, x, ps, st)

            if affine
                test_gradient_correctness_fdm((x, ps) -> sum(first(ln(x, ps, st))), x, ps;
                                              atol=1.0f-1, rtol=1.0f-1)
            else
                test_gradient_correctness_fdm(x -> sum(first(ln(x, ps, st))), x;
                                              atol=1.0f-1, rtol=1.0f-1)
            end

            for act in (sigmoid, tanh)
                ln = LayerNorm(bshape, act; affine)
                display(ln)
                ps, st = Lux.setup(rng, ln)

                @inferred first(ln(x, ps, st))
                y, st_ = ln(x, ps, st)

                run_JET_tests(ln, x, ps, st)

                if affine
                    test_gradient_correctness_fdm((x, ps) -> sum(first(ln(x, ps, st))), x,
                                                  ps; atol=1.0f-1, rtol=1.0f-1)
                else
                    test_gradient_correctness_fdm(x -> sum(first(ln(x, ps, st))), x;
                                                  atol=1.0f-1, rtol=1.0f-1)
                end
            end
        end
    end

    @testset "allow fast activation" begin
        layer = LayerNorm((3, 1), tanh)
        @test layer.activation == tanh_fast
        layer = LayerNorm((3, 1), tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end
end

@testset "InstanceNorm" begin
    for x in (randn(rng, Float32, 3, 3, 3, 2), randn(rng, Float32, 3, 3, 2),
              randn(rng, Float32, 3, 3, 3, 3, 2))
        for affine in (true, false)
            layer = InstanceNorm(3; affine)
            display(layer)
            ps, st = Lux.setup(rng, layer)

            @inferred first(layer(x, ps, st))
            y, st_ = layer(x, ps, st)

            run_JET_tests(layer, x, ps, st)

            if affine
                test_gradient_correctness_fdm((x, ps) -> sum(first(layer(x, ps, st))), x,
                                              ps; atol=1.0f-1, rtol=1.0f-1)
            else
                test_gradient_correctness_fdm(x -> sum(first(layer(x, ps, st))), x;
                                              atol=1.0f-1, rtol=1.0f-1)
            end

            for act in (sigmoid, tanh)
                layer = InstanceNorm(3, act; affine)
                display(layer)
                ps, st = Lux.setup(rng, layer)

                @inferred first(layer(x, ps, st))
                y, st_ = layer(x, ps, st)

                run_JET_tests(layer, x, ps, st)

                if affine
                    test_gradient_correctness_fdm((x, ps) -> sum(first(layer(x, ps, st))),
                                                  x, ps; atol=1.0f-1, rtol=1.0f-1)
                else
                    test_gradient_correctness_fdm(x -> sum(first(layer(x, ps, st))), x;
                                                  atol=1.0f-1, rtol=1.0f-1)
                end
            end
        end
    end

    @testset "allow fast activation" begin
        layer = InstanceNorm(3, tanh)
        @test layer.activation == tanh_fast
        layer = InstanceNorm(3, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end
end

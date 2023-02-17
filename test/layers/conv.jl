using Lux, NNlib, Random, Test, Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

include("../test_utils.jl")

@testset "Pooling" begin
    x = randn(rng, Float32, 10, 10, 3, 2)
    y = randn(rng, Float32, 20, 20, 3, 2)

    layer = AdaptiveMaxPool((5, 5))
    display(layer)
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == maxpool(x, PoolDims(x, 2))
    run_JET_tests(layer, x, ps, st)

    layer = AdaptiveMeanPool((5, 5))
    display(layer)
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == meanpool(x, PoolDims(x, 2))
    run_JET_tests(layer, x, ps, st)

    layer = AdaptiveMaxPool((10, 5))
    display(layer)
    ps, st = Lux.setup(rng, layer)

    @test layer(y, ps, st)[1] == maxpool(y, PoolDims(y, (2, 4)))
    run_JET_tests(layer, x, ps, st)

    layer = AdaptiveMeanPool((10, 5))
    display(layer)
    ps, st = Lux.setup(rng, layer)

    @test layer(y, ps, st)[1] == meanpool(y, PoolDims(y, (2, 4)))
    run_JET_tests(layer, x, ps, st)

    layer = GlobalMaxPool()
    display(layer)
    ps, st = Lux.setup(rng, layer)

    @test size(layer(x, ps, st)[1]) == (1, 1, 3, 2)
    run_JET_tests(layer, x, ps, st)

    layer = GlobalMeanPool()
    display(layer)
    ps, st = Lux.setup(rng, layer)

    @test size(layer(x, ps, st)[1]) == (1, 1, 3, 2)
    run_JET_tests(layer, x, ps, st)

    layer = MaxPool((2, 2))
    display(layer)
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == maxpool(x, PoolDims(x, 2))
    run_JET_tests(layer, x, ps, st)

    layer = MeanPool((2, 2))
    display(layer)
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == meanpool(x, PoolDims(x, 2))
    run_JET_tests(layer, x, ps, st)

    @testset "$ltype SamePad windowsize $k" for ltype in (MeanPool, MaxPool),
                                                k in ((1,), (2,), (3,), (4, 5), (6, 7, 8))

        x = ones(Float32, (k .+ 3)..., 1, 1)

        layer = ltype(k; pad=Lux.SamePad())
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1])[1:(end - 2)] == cld.(size(x)[1:(end - 2)], k)
        run_JET_tests(layer, x, ps, st)
    end
end

@testset "CNN" begin
    @testset "Grouped Conv" begin
        x = rand(rng, Float32, 4, 6, 1)
        layer = Conv((3,), 6 => 2; groups=2)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (3, 3, 2)
        @test size(layer(x, ps, st)[1]) == (2, 2, 1)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        x = rand(rng, Float32, 4, 4, 6, 1)
        layer = Conv((3, 3), 6 => 2; groups=2)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (3, 3, 3, 2)
        @test size(layer(x, ps, st)[1]) == (2, 2, 2, 1)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        x = rand(rng, Float32, 4, 4, 4, 6, 1)
        layer = Conv((3, 3, 3), 6 => 2; groups=2)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (3, 3, 3, 3, 2)
        @test size(layer(x, ps, st)[1]) == (2, 2, 2, 2, 1)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        # Test that we cannot ask for non-integer multiplication factors
        layer = Conv((2, 2), 3 => 10; groups=2)
        display(layer)
        @test_throws AssertionError Lux.setup(rng, layer)
        layer = Conv((2, 2), 2 => 9; groups=2)
        display(layer)
        @test_throws AssertionError Lux.setup(rng, layer)
    end

    @testset "Asymmetric Padding" begin
        layer = Conv((3, 3), 1 => 1, relu; pad=(0, 1, 1, 2))
        display(layer)
        x = ones(Float32, 28, 28, 1, 1)
        ps, st = Lux.setup(rng, layer)

        ps.weight .= 1.0
        ps.bias .= 0.0

        y_hat = layer(x, ps, st)[1][:, :, 1, 1]
        @test size(y_hat) == (27, 29)
        @test y_hat[1, 1] ≈ 6.0
        @test y_hat[2, 2] ≈ 9.0
        @test y_hat[end, 1] ≈ 4.0
        @test y_hat[1, end] ≈ 3.0
        @test y_hat[1, end - 1] ≈ 6.0
        @test y_hat[end, end] ≈ 2.0

        run_JET_tests(layer, x, ps, st)
    end

    @testset "Variable BitWidth Parameters" begin
        # https://github.com/FluxML/Flux.jl/issues/1421
        layer = Conv((5, 5), 10 => 20, identity; init_weight=Base.randn,
                     init_bias=(rng, dims...) -> randn(rng, Float16, dims...))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        @test ps.weight isa Array{Float64, 4}
        @test ps.bias isa Array{Float16, 4}
    end

    @testset "Depthwise Conv" begin
        x = randn(rng, Float32, 4, 4, 3, 2)

        layer = Conv((2, 2), 3 => 15; groups=3)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        @test Lux.parameterlength(layer) == Lux.parameterlength(ps)

        @test size(layer(x, ps, st)[1], 3) == 15
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        layer = Conv((2, 2), 3 => 9; groups=3)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1], 3) == 9
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        layer = Conv((2, 2), 3 => 9; groups=3, use_bias=false)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        @test Lux.parameterlength(layer) == Lux.parameterlength(ps)

        @test size(layer(x, ps, st)[1], 3) == 9
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        # Test that we cannot ask for non-integer multiplication factors
        layer = Conv((2, 2), 3 => 10; groups=3)
        display(layer)
        @test_throws AssertionError Lux.setup(rng, layer)
    end

    @testset "Conv SamePad kernelsize $k" for k in ((1,), (2,), (3,), (2, 3), (1, 2, 3))
        x = ones(Float32, (k .+ 3)..., 1, 1)

        layer = Conv(k, 1 => 1; pad=Lux.SamePad())
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1]) == size(x)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        layer = Conv(k, 1 => 1; pad=Lux.SamePad(), dilation=k .÷ 2)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1]) == size(x)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        stride = 3
        layer = Conv(k, 1 => 1; pad=Lux.SamePad(), stride=stride)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1])[1:(end - 2)] == cld.(size(x)[1:(end - 2)], stride)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end

    @testset "Conv with non quadratic window #700" begin
        x = zeros(Float32, 7, 7, 1, 1)
        x[4, 4, 1, 1] = 1

        layer = Conv((3, 3), 1 => 1)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        y = zeros(eltype(ps.weight), 5, 5, 1, 1)
        y[2:(end - 1), 2:(end - 1), 1, 1] = ps.weight
        @test y ≈ layer(x, ps, st)[1]
        run_JET_tests(layer, x, ps, st)

        layer = Conv((3, 1), 1 => 1)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        y = zeros(eltype(ps.weight), 5, 7, 1, 1)
        y[2:(end - 1), 4, 1, 1] = ps.weight
        @test y ≈ layer(x, ps, st)[1]
        run_JET_tests(layer, x, ps, st)

        layer = Conv((1, 3), 1 => 1)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        y = zeros(eltype(ps.weight), 7, 5, 1, 1)
        y[4, 2:(end - 1), 1, 1] = ps.weight
        @test y ≈ layer(x, ps, st)[1]
        run_JET_tests(layer, x, ps, st)

        layer = Conv((1, 3), 1 => 1; init_weight=Lux.glorot_normal)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        y = zeros(eltype(ps.weight), 7, 5, 1, 1)
        y[4, 2:(end - 1), 1, 1] = ps.weight
        @test y ≈ layer(x, ps, st)[1]
        run_JET_tests(layer, x, ps, st)
    end

    @testset "allow fast activation" begin
        layer = Conv((3, 3), 1 => 1, tanh)
        @test layer.activation == tanh_fast
        layer = Conv((3, 3), 1 => 1, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end

    # Deprecated Functionality (Remove in v0.5)
    @testset "Deprecations" begin
        @test_deprecated layer = Conv((3, 3), 1 => 1; bias=false)
        ps, st = Lux.setup(rng, layer)
        @test !hasproperty(ps, :bias)

        @test_throws ArgumentError layer=Conv((3, 3), 1 => 1; bias=false, use_bias=false)
    end
end

@testset "Upsample" begin
    @testset "Construction" begin
        @test_nowarn Upsample(:nearest; scale=2)
        @test_nowarn Upsample(:nearest; size=(64, 64))
        @test_nowarn Upsample(:bilinear; scale=2)
        @test_nowarn Upsample(:bilinear; size=(64, 64))
        @test_nowarn Upsample(:trilinear; scale=2)
        @test_nowarn Upsample(:trilinear; size=(64, 64))

        @test_throws ArgumentError Upsample(:linear; scale=2)
        @test_throws ArgumentError Upsample(:nearest; scale=2, size=(64, 64))
        @test_throws ArgumentError Upsample(:nearest)

        @test_nowarn Upsample(2)
        @test_nowarn Upsample(2, :nearest)
    end

    @testset "Size Correctness" begin
        # NNlib is checking algorithmic correctness. So we should just verify correct
        # function call
        modes = (:nearest, :bilinear, :trilinear)
        sizes = (nothing, (64, 64), (64, 32))
        scales = (nothing, 2, (2, 1))

        for mode in modes, xsize in sizes, scale in scales
            if !xor(isnothing(xsize), isnothing(scale))
                continue
            end
            layer = Upsample(mode; size=xsize, scale=scale)
            display(layer)
            ps, st = Lux.setup(rng, layer)
            x = zeros((32, 32, 3, 4))

            run_JET_tests(layer, x, ps, st)

            y, _ = layer(x, ps, st)
            if isnothing(scale)
                @test size(y)[1:2] == xsize
            else
                @test size(y)[1:2] == size(x)[1:2] .* scale
            end
            @test size(y)[3:4] == size(x)[3:4]
        end

        sizes = (nothing, (64, 64, 64), (64, 32, 128))
        scales = (nothing, 2, (2, 1, 1), (2, 2, 1))

        for mode in modes, xsize in sizes, scale in scales
            if !xor(isnothing(xsize), isnothing(scale))
                continue
            end
            layer = Upsample(mode; size=xsize, scale=scale)
            display(layer)
            ps, st = Lux.setup(rng, layer)
            x = zeros((32, 32, 32, 3, 4))

            run_JET_tests(layer, x, ps, st)

            y, _ = layer(x, ps, st)

            if isnothing(scale)
                @test size(y)[1:3] == xsize
            else
                @test size(y)[1:3] == size(x)[1:3] .* scale
            end
            @test size(y)[4:5] == size(x)[4:5]
        end
    end
end

@testset "PixelShuffle" begin
    layer = PixelShuffle(2)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    x = rand(rng, Float32, 3, 6, 3)

    y, st_ = layer(x, ps, st)
    @test y isa Array{Float32, 3}
    @test size(y) == (6, 3, 3)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1e-3, rtol=1e-3)

    layer = PixelShuffle(3)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    x = rand(Float32, 3, 4, 9, 3)

    y, st_ = layer(x, ps, st)
    @test y isa Array{Float32, 4}
    @test size(y) == (9, 12, 1, 3)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1e-3, rtol=1e-3)
end

@testset "CrossCor" begin
    @testset "Asymmetric Padding" begin
        layer = CrossCor((3, 3), 1 => 1, relu; pad=(0, 1, 1, 2))
        display(layer)
        x = ones(Float32, 28, 28, 1, 1)
        ps, st = Lux.setup(rng, layer)

        ps.weight .= 1.0
        ps.bias .= 0.0

        y_hat = layer(x, ps, st)[1][:, :, 1, 1]
        @test size(y_hat) == (27, 29)
        @test y_hat[1, 1] ≈ 6.0
        @test y_hat[2, 2] ≈ 9.0
        @test y_hat[end, 1] ≈ 4.0
        @test y_hat[1, end] ≈ 3.0
        @test y_hat[1, end - 1] ≈ 6.0
        @test y_hat[end, end] ≈ 2.0

        run_JET_tests(layer, x, ps, st)
    end

    @testset "Variable BitWidth Parameters" begin
        # https://github.com/FluxML/Flux.jl/issues/1421
        layer = CrossCor((5, 5), 10 => 20, identity; init_weight=Base.randn,
                         init_bias=(rng, dims...) -> randn(rng, Float16, dims...))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        @test ps.weight isa Array{Float64, 4}
        @test ps.bias isa Array{Float16, 4}
    end

    @testset "CrossCor SamePad kernelsize $k" for k in ((1,), (2,), (3,), (2, 3), (1, 2, 3))
        x = ones(Float32, (k .+ 3)..., 1, 1)

        layer = CrossCor(k, 1 => 1; pad=Lux.SamePad())
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1]) == size(x)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        layer = CrossCor(k, 1 => 1; pad=Lux.SamePad(), dilation=k .÷ 2)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1]) == size(x)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        stride = 3
        layer = CrossCor(k, 1 => 1; pad=Lux.SamePad(), stride=stride)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1])[1:(end - 2)] == cld.(size(x)[1:(end - 2)], stride)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end

    @testset "allow fast activation" begin
        layer = CrossCor((3, 3), 1 => 1, tanh)
        @test layer.activation == tanh_fast
        layer = CrossCor((3, 3), 1 => 1, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end
end

@testset "ConvTranspose" begin
    x = randn(Float32, 5, 5, 1, 1)
    layer = Conv((3, 3), 1 => 1)
    ps, st = Lux.setup(rng, layer)
    y = layer(x, ps, st)[1]

    layer = ConvTranspose((3, 3), 1 => 1)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    run_JET_tests(layer, y, ps, st; opt_broken=true)
    @static if VERSION >= v"1.7"
        # Inference broken in v1.6
        @inferred layer(y, ps, st)
    end
    x_hat1 = layer(y, ps, st)[1]

    layer = ConvTranspose((3, 3), 1 => 1; use_bias=false)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    run_JET_tests(layer, y, ps, st; opt_broken=true)
    @static if VERSION >= v"1.7"
        # Inference broken in v1.6
        @inferred layer(y, ps, st)
    end
    x_hat2 = layer(y, ps, st)[1]

    @test size(x_hat1) == size(x_hat2) == size(x)

    layer = ConvTranspose((3, 3), 1 => 1)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    x = rand(Float32, 5, 5, 1, 1)
    run_JET_tests(layer, x, ps, st; opt_broken=true)
    @static if VERSION >= v"1.7"
        # Inference broken in v1.6
        @inferred layer(x, ps, st)
    end
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    x = rand(Float32, 5, 5, 2, 4)
    layer = ConvTranspose((3, 3), 2 => 3)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    run_JET_tests(layer, x, ps, st; opt_broken=true)
    @static if VERSION >= v"1.7"
        # Inference broken in v1.6
        @inferred layer(x, ps, st)
    end
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    # test ConvTranspose supports groups argument
    x = randn(Float32, 10, 10, 2, 3)
    layer1 = ConvTranspose((3, 3), 2 => 4; pad=SamePad())
    display(layer1)
    ps1, st1 = Lux.setup(rng, layer1)
    @test size(ps1.weight) == (3, 3, 4, 2)
    @test size(layer1(x, ps1, st1)[1]) == (10, 10, 4, 3)

    layer2 = ConvTranspose((3, 3), 2 => 4; groups=2, pad=SamePad())
    display(layer2)
    ps2, st2 = Lux.setup(rng, layer2)
    @test size(ps2.weight) == (3, 3, 2, 2)
    @test size(layer1(x, ps1, st1)[1]) == size(layer2(x, ps2, st2)[1])

    test_gradient_correctness_fdm((x, ps) -> sum(layer1(x, ps, st1)[1]), x, ps1;
                                  atol=1.0f-3, rtol=1.0f-3)
    test_gradient_correctness_fdm((x, ps) -> sum(layer2(x, ps, st2)[1]), x, ps2;
                                  atol=1.0f-3, rtol=1.0f-3)

    x = randn(Float32, 10, 2, 1)
    layer = ConvTranspose((3,), 2 => 4; pad=SamePad(), groups=2)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    run_JET_tests(layer, x, ps, st; opt_broken=true)
    @test size(layer(x, ps, st)[1]) == (10, 4, 1)
    @test length(ps.weight) == 3 * (2 * 4) / 2
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    x = randn(Float32, 10, 11, 4, 2)
    layer = ConvTranspose((3, 5), 4 => 4; pad=SamePad(), groups=4)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    run_JET_tests(layer, x, ps, st; opt_broken=true)
    @test size(layer(x, ps, st)[1]) == (10, 11, 4, 2)
    @test length(ps.weight) == (3 * 5) * (4 * 4) / 4
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    x = randn(Float32, 10, 11, 4, 2)
    layer = ConvTranspose((3, 5), 4 => 4, tanh; pad=SamePad(), groups=4)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    run_JET_tests(layer, x, ps, st; opt_broken=true)
    @test size(layer(x, ps, st)[1]) == (10, 11, 4, 2)
    @test length(ps.weight) == (3 * 5) * (4 * 4) / 4
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    x = randn(Float32, 10, 11, 12, 3, 2)
    layer = ConvTranspose((3, 5, 3), 3 => 6; pad=SamePad(), groups=3)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    run_JET_tests(layer, x, ps, st; opt_broken=true)
    @test size(layer(x, ps, st)[1]) == (10, 11, 12, 6, 2)
    @test length(ps.weight) == (3 * 5 * 3) * (3 * 6) / 3

    x = randn(Float32, 10, 11, 12, 3, 2)
    layer = ConvTranspose((3, 5, 3), 3 => 6, tanh; pad=SamePad(), groups=3)
    display(layer)
    ps, st = Lux.setup(rng, layer)
    run_JET_tests(layer, x, ps, st; opt_broken=true)
    @test size(layer(x, ps, st)[1]) == (10, 11, 12, 6, 2)
    @test length(ps.weight) == (3 * 5 * 3) * (3 * 6) / 3

    @test occursin("groups=2", sprint(show, ConvTranspose((3, 3), 2 => 4; groups=2)))
    @test occursin("2 => 4", sprint(show, ConvTranspose((3, 3), 2 => 4; groups=2)))
end

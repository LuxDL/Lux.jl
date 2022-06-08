using Lux, NNlib, Random, Test, Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

include("../utils.jl")

@testset "Pooling" begin
    x = randn(rng, Float32, 10, 10, 3, 2)
    y = randn(rng, Float32, 20, 20, 3, 2)

    layer = AdaptiveMaxPool((5, 5))
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == maxpool(x, PoolDims(x, 2))
    run_JET_tests(layer, x, ps, st)

    layer = AdaptiveMeanPool((5, 5))
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == meanpool(x, PoolDims(x, 2))
    run_JET_tests(layer, x, ps, st)

    layer = AdaptiveMaxPool((10, 5))
    ps, st = Lux.setup(rng, layer)

    @test layer(y, ps, st)[1] == maxpool(y, PoolDims(y, (2, 4)))
    run_JET_tests(layer, x, ps, st)

    layer = AdaptiveMeanPool((10, 5))
    ps, st = Lux.setup(rng, layer)

    @test layer(y, ps, st)[1] == meanpool(y, PoolDims(y, (2, 4)))
    run_JET_tests(layer, x, ps, st)

    layer = GlobalMaxPool()
    ps, st = Lux.setup(rng, layer)

    @test size(layer(x, ps, st)[1]) == (1, 1, 3, 2)
    run_JET_tests(layer, x, ps, st)

    layer = GlobalMeanPool()
    ps, st = Lux.setup(rng, layer)

    @test size(layer(x, ps, st)[1]) == (1, 1, 3, 2)
    run_JET_tests(layer, x, ps, st)

    layer = MaxPool((2, 2))
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == maxpool(x, PoolDims(x, 2))
    run_JET_tests(layer, x, ps, st)

    layer = MeanPool((2, 2))
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == meanpool(x, PoolDims(x, 2))
    run_JET_tests(layer, x, ps, st)

    @testset "$ltype SamePad windowsize $k" for ltype in (MeanPool, MaxPool),
                                                k in ((1,), (2,), (3,), (4, 5), (6, 7, 8))

        x = ones(Float32, (k .+ 3)..., 1, 1)

        layer = ltype(k; pad=Lux.SamePad())
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1])[1:(end - 2)] == cld.(size(x)[1:(end - 2)], k)
        run_JET_tests(layer, x, ps, st; call_broken=length(k) == 1)
    end
end

@testset "CNN" begin
    @testset "Grouped Conv" begin
        x = rand(rng, Float32, 4, 6, 1)
        layer = Conv((3,), 6 => 2; groups=2)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (3, 3, 2)
        @test size(layer(x, ps, st)[1]) == (2, 2, 1)
        run_JET_tests(layer, x, ps, st; call_broken=true)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        x = rand(rng, Float32, 4, 4, 6, 1)
        layer = Conv((3, 3), 6 => 2; groups=2)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (3, 3, 3, 2)
        @test size(layer(x, ps, st)[1]) == (2, 2, 2, 1)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        x = rand(rng, Float32, 4, 4, 4, 6, 1)
        layer = Conv((3, 3, 3), 6 => 2; groups=2)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (3, 3, 3, 3, 2)
        @test size(layer(x, ps, st)[1]) == (2, 2, 2, 2, 1)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        # Test that we cannot ask for non-integer multiplication factors
        layer = Conv((2, 2), 3 => 10, groups=2)
        @test_throws AssertionError Lux.setup(rng, layer)
        layer = Conv((2, 2), 2 => 9, groups=2)
        @test_throws AssertionError Lux.setup(rng, layer)
    end

    @testset "Asymmetric Padding" begin
        layer = Conv((3, 3), 1 => 1, relu; pad=(0, 1, 1, 2))
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
        layer = Conv((5, 5), 10 => 20, identity; init_weight=Base.randn)
        ps, st = Lux.setup(rng, layer)
        @test ps.bias isa Array{Float64, 4}
    end

    @testset "Depthwise Conv" begin
        x = randn(rng, Float32, 4, 4, 3, 2)

        layer = Conv((2, 2), 3 => 15; groups=3)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1], 3) == 15
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        layer = Conv((2, 2), 3 => 9; groups=3)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1], 3) == 9
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        layer = Conv((2, 2), 3 => 9; groups=3, bias=false)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1], 3) == 9
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        # Test that we cannot ask for non-integer multiplication factors
        layer = Conv((2, 2), 3 => 10; groups=3)
        @test_throws AssertionError Lux.setup(rng, layer)
    end

    @testset "Conv SamePad kernelsize $k" for k in ((1,), (2,), (3,), (2, 3), (1, 2, 3))
        x = ones(Float32, (k .+ 3)..., 1, 1)

        layer = Conv(k, 1 => 1; pad=Lux.SamePad())
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1]) == size(x)
        run_JET_tests(layer, x, ps, st; call_broken=length(k) == 1)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        layer = Conv(k, 1 => 1; pad=Lux.SamePad(), dilation=k .÷ 2)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1]) == size(x)
        run_JET_tests(layer, x, ps, st; call_broken=length(k) == 1)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        stride = 3
        layer = Conv(k, 1 => 1; pad=Lux.SamePad(), stride=stride)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1])[1:(end - 2)] == cld.(size(x)[1:(end - 2)], stride)
        run_JET_tests(layer, x, ps, st; call_broken=length(k) == 1)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end

    @testset "Conv with non quadratic window #700" begin
        x = zeros(Float32, 7, 7, 1, 1)
        x[4, 4, 1, 1] = 1

        layer = Conv((3, 3), 1 => 1)
        ps, st = Lux.setup(rng, layer)

        y = zeros(eltype(ps.weight), 5, 5, 1, 1)
        y[2:(end - 1), 2:(end - 1), 1, 1] = ps.weight
        @test y ≈ layer(x, ps, st)[1]
        run_JET_tests(layer, x, ps, st)

        layer = Conv((3, 1), 1 => 1)
        ps, st = Lux.setup(rng, layer)

        y = zeros(eltype(ps.weight), 5, 7, 1, 1)
        y[2:(end - 1), 4, 1, 1] = ps.weight
        @test y ≈ layer(x, ps, st)[1]
        run_JET_tests(layer, x, ps, st)

        layer = Conv((1, 3), 1 => 1)
        ps, st = Lux.setup(rng, layer)

        y = zeros(eltype(ps.weight), 7, 5, 1, 1)
        y[4, 2:(end - 1), 1, 1] = ps.weight
        @test y ≈ layer(x, ps, st)[1]
        run_JET_tests(layer, x, ps, st)
    end
end

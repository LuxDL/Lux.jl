using Lux, NNlib, Random, Test

include("../test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "Miscellaneous Layers" begin
    @testset "Reshape Layer" begin
        layer = ReshapeLayer((2, 3))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 3)

        @test size(layer(x, ps, st)[1]) == (2, 3, 3)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)
    end

    @testset "Flatten Layer" begin
        layer = FlattenLayer()
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 3, 2)

        @test size(layer(x, ps, st)[1]) == (18, 2)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)
    end

    @testset "NoOpLayer" begin
        layer = NoOpLayer()
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = (x=2, b=5) # Something totally arbitrary

        @test layer(x, ps, st)[1] == x
        run_JET_tests(layer, x, ps, st)

        x = randn(rng, 6, 3)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)
    end

    @testset "SelectDim Layer" begin
        layer = SelectDim(3, 1)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 4, 3, 2)

        @test size(layer(x, ps, st)[1]) == (6, 4, 2)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)
    end

    @testset "WrappedFunction" begin
        layer = WrappedFunction(x -> x .* x)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 4, 3, 2)

        @test layer(x, ps, st)[1] == x .* x
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)
    end

    @testset "ActivationFunction" begin
        layer = ActivationFunction(tanh)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 4, 3, 2)

        @test layer(x, ps, st)[1] == tanh.(x)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)
    end
end

@testset "Dense" begin
    @testset "constructors" begin
        layer = Dense(10, 100)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (100, 10)
        @test size(ps.bias) == (100, 1)
        @test layer.activation == identity

        layer = Dense(10, 100, relu; use_bias=false)
        ps, st = Lux.setup(rng, layer)

        @test !haskey(ps, :bias)
        @test layer.activation == relu
    end

    @testset "allow fast activation" begin
        layer = Dense(10, 10, tanh)
        @test layer.activation == tanh_fast
        layer = Dense(10, 10, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end

    @testset "dimensions" begin
        layer = Dense(10, 5)
        ps, st = Lux.setup(rng, layer)

        @test size(first(Lux.apply(layer, randn(10), ps, st))) == (5,)
        @test size(first(Lux.apply(layer, randn(10, 2), ps, st))) == (5, 2)
    end

    @testset "zeros" begin
        @test begin
            layer = Dense(10, 1, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 1), Lux.setup(rng, layer)...))
        end == 10 * ones(1, 1)

        @test begin
            layer = Dense(10, 1, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 2), Lux.setup(rng, layer)...))
        end == 10 * ones(1, 2)

        @test begin
            layer = Dense(10, 2, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 1), Lux.setup(rng, layer)...))
        end == 10 * ones(2, 1)

        @test begin
            layer = Dense(10, 2, identity; init_weight=ones)
            first(Lux.apply(layer, [ones(10, 1) 2 * ones(10, 1)], Lux.setup(rng, layer)...))
        end == [10 20; 10 20]

        @test begin
            layer = Dense(10, 2, identity; init_weight=ones, bias=false)
            first(Lux.apply(layer, [ones(10, 1) 2 * ones(10, 1)], Lux.setup(rng, layer)...))
        end == [10 20; 10 20]
    end

    # Deprecated Functionality (Remove in v0.5)
    @testset "Deprecations" begin
        @test_deprecated Dense(10, 100, relu; bias=false)
        @test_deprecated Dense(10, 100, relu; bias=true)
        @test_throws ArgumentError Dense(10, 100, relu; bias=false, use_bias=false)
    end
end

@testset "Scale" begin
    @testset "constructors" begin
        layer = Scale(10, 100)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (10, 100)
        @test size(ps.bias) == (10, 100)
        @test layer.activation == identity

        layer = Scale(10, 100, relu; use_bias=false)
        ps, st = Lux.setup(rng, layer)

        @test !haskey(ps, :bias)
        @test layer.activation == relu
    end

    @testset "allow fast activation" begin
        layer = Scale(10, 5, tanh)
        @test layer.activation == tanh_fast
        layer = Scale(10, 5, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end

    @testset "dimensions" begin
        layer = Scale(10, 5)
        ps, st = Lux.setup(rng, layer)

        @test size(first(Lux.apply(layer, randn(10), ps, st))) == (10, 5)
        @test size(first(Lux.apply(layer, randn(10, 5, 2), ps, st))) == (10, 5, 2)
    end

    @testset "zeros" begin
        @test begin
            layer = Scale(10, 1, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 1), Lux.setup(rng, layer)...))
        end == ones(10, 1)

        @test begin
            layer = Scale(10, 1, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 2), Lux.setup(rng, layer)...))
        end == ones(10, 2)

        @test begin
            layer = Scale(2, identity; init_weight=ones, init_bias=ones)
            first(Lux.apply(layer, [1 2; 3 4], Lux.setup(rng, layer)...))
        end == [2.0 3.0; 4.0 5.0]

        @test begin
            layer = Scale(2, tanh; bias=false, init_weight=zeros)
            first(Lux.apply(layer, [1 2; 3 4], Lux.setup(rng, layer)...))
        end == zeros(2, 2)
    end

    # Deprecated Functionality (Remove in v0.5)
    @testset "Deprecations" begin
        @test_deprecated Scale(10, 100, relu; bias=false)
        @test_deprecated Scale(10, 100, relu; bias=true)
        @test_throws ArgumentError Scale(10, 100, relu; bias=false, use_bias=false)
    end
end

@testset "Bilinear" begin
    @testset "SkipConnection recombinator" begin
        d = Dense(2 => 2)
        display(d)
        b = Bilinear((2, 2) => 3)
        display(b)
        layer = SkipConnection(d, b)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, Float32, 2, 1)

        @test size(layer(x, ps, st)[1]) == (3, 1)
        @inferred layer(x, ps, st)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        d = Dense(2 => 2)
        display(d)
        b = Bilinear((2, 2) => 3; use_bias=false)
        display(b)
        layer = SkipConnection(d, b)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, Float32, 2, 1)

        @test size(layer(x, ps, st)[1]) == (3, 1)
        @inferred layer(x, ps, st)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end

    @testset "Two-streams zero sum" begin
        x = zeros(Float32, 2, 1)
        y = zeros(Float32, 1, 1)
        layer = Bilinear((2, 1) => 3)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer((x, y), ps, st)[1]) == (3, 1)
        @test sum(abs2, layer((x, y), ps, st)[1]) == 0.0f0
        @inferred layer((x, y), ps, st)
        run_JET_tests(layer, (x, y), ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), (x, y), ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end

    @testset "Inner interactions" begin
        x = randn(Float32, 2, 1)
        layer = Bilinear((2, 2) => 3)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1]) == (3, 1)
        @inferred layer(x, ps, st)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)

        x = randn(Float32, 2, 1)
        layer = Bilinear(2 => 3)
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1]) == (3, 1)
        @inferred layer(x, ps, st)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end
end

@testset "Embedding" begin
    vocab_size, embed_size = 10, 4
    layer = Embedding(vocab_size => embed_size)
    display(layer)
    ps, st = Lux.setup(rng, layer)

    @test size(ps.weight) == (embed_size, vocab_size)

    x = rand(1:vocab_size, 1)[1]
    y, st_ = layer(x, ps, st)
    @test size(layer(x, ps, st)[1]) == (embed_size,)
    @test y == ps.weight[:, x]
    @inferred layer(x, ps, st)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm(ps -> sum(layer(x, ps, st)[1]), ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    x = rand(1:vocab_size, 3)
    y, st_ = layer(x, ps, st)
    @test y isa Matrix{Float32}
    @test y == ps.weight[:, x]
    @inferred layer(x, ps, st)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm(ps -> sum(layer(x, ps, st)[1]), ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    x = rand(1:vocab_size, 3, 4)
    y, st_ = layer(x, ps, st)
    @test y isa Array{Float32, 3}
    @test size(y) == (embed_size, 3, 4)
    @inferred layer(x, ps, st)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm(ps -> sum(layer(x, ps, st)[1]), ps; atol=1.0f-3,
                                  rtol=1.0f-3)
end

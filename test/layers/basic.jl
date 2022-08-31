using Lux, NNlib, Random, Test

include("../test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "Miscellaneous Layers" begin
  @testset "Reshape Layer" begin
    layer = ReshapeLayer((2, 3))
    println(layer)
    ps, st = Lux.setup(rng, layer)
    x = randn(rng, 6, 3)

    @test size(layer(x, ps, st)[1]) == (2, 3, 3)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                  rtol=1.0f-3)
  end

  @testset "Flatten Layer" begin
    layer = FlattenLayer()
    println(layer)
    ps, st = Lux.setup(rng, layer)
    x = randn(rng, 6, 3, 2)

    @test size(layer(x, ps, st)[1]) == (18, 2)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                  rtol=1.0f-3)
  end

  @testset "NoOpLayer" begin
    layer = NoOpLayer()
    println(layer)
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
    println(layer)
    ps, st = Lux.setup(rng, layer)
    x = randn(rng, 6, 4, 3, 2)

    @test size(layer(x, ps, st)[1]) == (6, 4, 2)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                  rtol=1.0f-3)
  end

  @testset "WrappedFunction" begin
    layer = WrappedFunction(x -> x .* x)
    println(layer)
    ps, st = Lux.setup(rng, layer)
    x = randn(rng, 6, 4, 3, 2)

    @test layer(x, ps, st)[1] == x .* x
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                  rtol=1.0f-3)
  end

  @testset "ActivationFunction" begin
    layer = ActivationFunction(tanh)
    println(layer)
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

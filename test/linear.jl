@testset "constructors" begin
    layer = Dense(10, 100)
    ps, st = setup(MersenneTwister(0), layer)

    @test size(ps.weight) == (100, 10)
    @test size(ps.bias) == (100, 1)
    @test layer.λ == identity

    layer = Dense(10, 100, relu; bias=false)
    ps, st = setup(MersenneTwister(0), layer)

    @test !haskey(ps, :bias)
    @test layer.λ == relu
end

@testset "dimensions" begin
    layer = Dense(10, 5)
    ps, st = setup(MersenneTwister(0), layer)

    @test length(first(apply(layer, randn(10), ps, st))) == 5
    @test_throws DimensionMismatch first(apply(layer, randn(1), ps, st))
    @test_throws MethodError first(apply(layer, 1, ps, st))
    @test size(first(apply(layer, randn(10), ps, st))) == (5,)
    @test size(first(apply(layer, randn(10, 2), ps, st))) == (5, 2)
    @test size(first(apply(layer, randn(10, 2, 3), ps, st))) == (5, 2, 3)
    @test size(first(apply(layer, randn(10, 2, 3, 4), ps, st))) == (5, 2, 3, 4)
    @test_throws DimensionMismatch first(apply(layer, randn(11, 2, 3), ps, st))
end

@testset "zeros" begin
    @test begin
        layer = Dense(10, 1, identity; initW=ones)
        first(apply(layer, ones(10, 1), setup(MersenneTwister(0), layer)...))
    end == 10 * ones(1, 1)

    @test begin
        layer = Dense(10, 1, identity; initW=ones)
        first(apply(layer, ones(10, 2), setup(MersenneTwister(0), layer)...))
    end == 10 * ones(1, 2)

    @test begin
        layer = Dense(10, 2, identity; initW=ones)
        first(apply(layer, ones(10, 1), setup(MersenneTwister(0), layer)...))
    end == 10 * ones(2, 1)

    @test begin
        layer = Dense(10, 2, identity; initW=ones)
        first(apply(layer, [ones(10, 1) 2 * ones(10, 1)], setup(MersenneTwister(0), layer)...))
    end == [10 20; 10 20]

    @test begin
        layer = Dense(10, 2, identity; initW=ones, bias=false)
        first(apply(layer, [ones(10, 1) 2 * ones(10, 1)], setup(MersenneTwister(0), layer)...))
    end == [10 20; 10 20]
end

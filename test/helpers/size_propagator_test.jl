@testitem "Size Propagator" setup=[SharedTestSetup] tags=[:helpers] begin
    rng = StableRNG(12345)

    @testset "Simple Chain (LeNet)" begin
        lenet = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
            Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(),
            Dense(256 => 120, relu), Dense(120 => 84, relu), Dense(84 => 10))
        ps, st = Lux.setup(rng, lenet)

        @test Lux.compute_output_size(lenet, (28, 28, 1, 3), ps, st) == (10,)
        @test Lux.compute_output_size(lenet, (28, 28, 1, 12), ps, st) == (10,)
    end

    @testset "Chain with BatchNorm" begin
        lenet = Chain(Conv((5, 5), 1 => 6, relu), BatchNorm(6, relu), MaxPool((2, 2)),
            Conv((5, 5), 6 => 16, relu), BatchNorm(16, relu),
            MaxPool((2, 2)), FlattenLayer(), Dense(256 => 120, relu),
            BatchNorm(120, relu), Dense(120 => 84, relu), Dropout(0.5f0),
            BatchNorm(84, relu), Dense(84 => 10), BatchNorm(10, relu))
        ps, st = Lux.setup(rng, lenet)

        @test Lux.compute_output_size(lenet, (28, 28, 1, 3), ps, st) == (10,)
        @test Lux.compute_output_size(lenet, (28, 28, 1, 12), ps, st) == (10,)
    end

    @testset "Normalization Layers" begin
        layer = BatchNorm(3, relu)
        ps, st = Lux.setup(rng, layer)

        @test Lux.compute_output_size(layer, (4, 4, 3, 2), ps, st) == (4, 4, 3)
        @test Lux.compute_output_size(layer, (3, 3), ps, st) == (3,)

        layer = GroupNorm(6, 3, relu)
        ps, st = Lux.setup(rng, layer)

        @test Lux.compute_output_size(layer, (4, 4, 6, 2), ps, st) == (4, 4, 6)
        @test Lux.compute_output_size(layer, (6, 3), ps, st) == (6,)

        layer = InstanceNorm(3, relu)
        ps, st = Lux.setup(rng, layer)

        @test Lux.compute_output_size(layer, (4, 4, 3, 2), ps, st) == (4, 4, 3)
        @test Lux.compute_output_size(layer, (4, 3, 2), ps, st) == (4, 3)

        layer = LayerNorm((2, 1, 3), relu)
        ps, st = Lux.setup(rng, layer)

        @test Lux.compute_output_size(layer, (2, 4, 3, 2), ps, st) == (2, 4, 3)
        @test Lux.compute_output_size(layer, (2, 1, 3, 3), ps, st) == (2, 1, 3)
    end
end

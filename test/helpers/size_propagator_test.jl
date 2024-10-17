@testitem "Size Propagator" setup=[SharedTestSetup] tags=[:helpers] begin
    rng=StableRNG(12345)

    @testset "Simple Chain (LeNet)" begin
        lenet = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
            Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
            Dense(256 => 120, relu), Dense(120 => 84, relu), Dense(84 => 10))

        for x in (randn(rng, Float32, 28, 28, 1, 3), randn(rng, Float32, 28, 28, 1, 12))
            @test Lux.outputsize(lenet, x, rng) == (10,)
        end
    end

    @testset "Chain with BatchNorm" begin
        lenet = Chain(Conv((5, 5), 1 => 6, relu), BatchNorm(6, relu), MaxPool((2, 2)),
            Conv((5, 5), 6 => 16, relu), BatchNorm(16, relu),
            MaxPool((2, 2)), FlattenLayer(3), Dense(256 => 120, relu),
            BatchNorm(120, relu), Dense(120 => 84, relu), Dropout(0.5f0),
            BatchNorm(84, relu), Dense(84 => 10), BatchNorm(10, relu))

        for x in (randn(rng, Float32, 28, 28, 1, 3), randn(rng, Float32, 28, 28, 1, 12))
            @test Lux.outputsize(lenet, x, rng) == (10,)
        end
    end

    norm_layer=[
        (BatchNorm(3, relu), [randn(rng, Float32, 4, 4, 3, 2), randn(rng, Float32, 3, 3)]),
        (GroupNorm(6, 3, relu),
            [randn(rng, Float32, 4, 4, 6, 2), randn(rng, Float32, 6, 3)]),
        (InstanceNorm(3, relu),
            [randn(rng, Float32, 4, 4, 3, 2), randn(rng, Float32, 4, 3, 2)]),
        (LayerNorm((2, 1, 3), relu),
            [randn(rng, Float32, 2, 4, 3, 2), randn(rng, Float32, 2, 1, 3, 3)])]

    @testset "Normalization: $(nameof(typeof(layer)))" for (layer, xs) in norm_layer
        for x in xs
            @test Lux.outputsize(layer, x, rng) == size(x)[1:(end - 1)]
        end
    end
end

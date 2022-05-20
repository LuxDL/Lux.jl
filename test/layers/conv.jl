using JET, Lux, NNlib, Random, Test, Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

include("../utils.jl")

@testset "Pooling" begin
    x = randn(rng, Float32, 10, 10, 3, 2)
    y = randn(rng, Float32, 20, 20, 3, 2)

    layer = AdaptiveMaxPool((5, 5))
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == maxpool(x, PoolDims(x, 2))
    @test_call layer(x, ps, st)
    @test_opt target_modules = (Lux,) layer(x, ps, st)

    layer = AdaptiveMeanPool((5, 5))
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == meanpool(x, PoolDims(x, 2))
    @test_call layer(x, ps, st)
    @test_opt target_modules = (Lux,) layer(x, ps, st)

    layer = AdaptiveMaxPool((10, 5))
    ps, st = Lux.setup(rng, layer)

    @test layer(y, ps, st)[1] == maxpool(y, PoolDims(y, (2, 4)))
    @test_call layer(y, ps, st)
    @test_opt target_modules = (Lux,) layer(y, ps, st)

    layer = AdaptiveMeanPool((10, 5))
    ps, st = Lux.setup(rng, layer)

    @test layer(y, ps, st)[1] == meanpool(y, PoolDims(y, (2, 4)))
    @test_call layer(y, ps, st)
    @test_opt target_modules = (Lux,) layer(y, ps, st)

    layer = GlobalMaxPool()
    ps, st = Lux.setup(rng, layer)

    @test size(layer(x, ps, st)[1]) == (1, 1, 3, 2)
    @test_call layer(x, ps, st)
    @test_opt target_modules = (Lux,) layer(x, ps, st)

    layer = GlobalMeanPool()
    ps, st = Lux.setup(rng, layer)

    @test size(layer(x, ps, st)[1]) == (1, 1, 3, 2)
    @test_call layer(x, ps, st)
    @test_opt target_modules = (Lux,) layer(x, ps, st)

    layer = MaxPool((2, 2))
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == maxpool(x, PoolDims(x, 2))
    @test_call layer(x, ps, st)
    @test_opt target_modules = (Lux,) layer(x, ps, st)

    layer = MeanPool((2, 2))
    ps, st = Lux.setup(rng, layer)

    @test layer(x, ps, st)[1] == meanpool(x, PoolDims(x, 2))
    @test_call layer(x, ps, st)
    @test_opt target_modules = (Lux,) layer(x, ps, st)
end



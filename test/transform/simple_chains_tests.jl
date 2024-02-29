@testitem "ToSimpleChainsAdaptor tests" setup=[SharedTestSetup] begin
    import SimpleChains: static

    lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
        Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)))

    adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))

    simple_chains_model = adaptor(lux_model)

    ps, st = Lux.setup(Random.default_rng(), simple_chains_model)

    x = randn(Float32, 28, 28, 1, 1)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 1)

    gs = Zygote.gradient((x, p) -> sum(first(simple_chains_model(x, p, st))), x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    x = randn(Float32, 28, 28, 1, 15)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 15)

    gs = Zygote.gradient((x, p) -> sum(first(simple_chains_model(x, p, st))), x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    lux_model = Chain(
        FlattenLayer(3), Dense(784 => 20, tanh), Dropout(0.5), Dense(20 => 10))

    adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))

    simple_chains_model = adaptor(lux_model)

    ps, st = Lux.setup(Random.default_rng(), simple_chains_model)

    x = randn(Float32, 28, 28, 1, 1)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 1)

    gs = Zygote.gradient((x, p) -> sum(first(simple_chains_model(x, p, st))), x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    x = randn(Float32, 28, 28, 1, 15)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 15)

    gs = Zygote.gradient((x, p) -> sum(first(simple_chains_model(x, p, st))), x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)
end

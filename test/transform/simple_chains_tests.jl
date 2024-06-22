@testitem "ToSimpleChainsAdaptor" setup=[SharedTestSetup] tags=[:others] begin
    import SimpleChains: static

    lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
        Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)))

    adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))

    simple_chains_model = adaptor(lux_model)

    rng = Random.Xoshiro()
    ps_rng = []
    for i in 1:2
        ps, st = Lux.setup(Lux.replicate(rng), simple_chains_model)
        push!(ps_rng, ps)
    end
    @test allequal(ps_rng)

    ps, st = Lux.setup(Random.default_rng(), simple_chains_model)

    x = randn(Float32, 28, 28, 1, 1)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 1)

    __f = (x, p) -> sum(first(simple_chains_model(x, p, st)))

    gs = Zygote.gradient(__f, x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3

    x = randn(Float32, 28, 28, 1, 15)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 15)

    __f = (x, p) -> sum(first(simple_chains_model(x, p, st)))

    gs = Zygote.gradient(__f, x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3

    @testset "Array Output" begin
        adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)), true)
        simple_chains_model = adaptor(lux_model)

        ps, st = Lux.setup(Random.default_rng(), simple_chains_model)
        x = randn(Float32, 28, 28, 1, 1)

        @test size(first(simple_chains_model(x, ps, st))) == (10, 1)
        @test first(simple_chains_model(x, ps, st)) isa Array
    end

    lux_model = Chain(
        FlattenLayer(3), Dense(784 => 20, tanh), Dropout(0.5), Dense(20 => 10))

    adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))

    simple_chains_model = adaptor(lux_model)

    ps, st = Lux.setup(Random.default_rng(), simple_chains_model)

    x = randn(Float32, 28, 28, 1, 1)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 1)

    __f = (x, p) -> sum(first(simple_chains_model(x, p, st)))

    gs = Zygote.gradient(__f, x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    x = randn(Float32, 28, 28, 1, 15)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 15)

    __f = (x, p) -> sum(first(simple_chains_model(x, p, st)))

    gs = Zygote.gradient(__f, x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    @testset "Single Layer Conversion: LuxDL/Lux.jl#545 & LuxDL/Lux.jl#551" begin
        lux_model = Dense(10 => 5)

        for dims in (static(10), (static(10),))
            adaptor = ToSimpleChainsAdaptor(dims)

            simple_chains_model = @test_warn "The model provided is not a `Chain`. Trying to wrap it into a `Chain` but this might fail. Please consider using `Chain` directly (potentially with `disable_optimizations = true`)." adaptor(lux_model)

            ps, st = Lux.setup(Random.default_rng(), simple_chains_model)

            x = randn(Float32, 10, 3)
            @test size(first(simple_chains_model(x, ps, st))) == (5, 3)

            __f = (x, p) -> sum(first(simple_chains_model(x, p, st)))

            gs = Zygote.gradient(__f, x, ps)
            @test size(gs[1]) == size(x)
            @test length(gs[2].params) == length(ps.params)

            @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3
        end
    end

    @test_throws ArgumentError ToSimpleChainsAdaptor((10, 10, 1))
    @test_throws ArgumentError ToSimpleChainsAdaptor(1)

    # Failures
    @test_throws Lux.SimpleChainsModelConversionError adaptor(Conv(
        (1, 1), 2 => 3; stride=(5, 5)))
    @test_throws Lux.SimpleChainsModelConversionError adaptor(Dropout(0.2f0; dims=1))
    @test_throws Lux.SimpleChainsModelConversionError adaptor(FlattenLayer())
    @test_throws Lux.SimpleChainsModelConversionError adaptor(MaxPool(
        (2, 2); stride=(1, 1)))
    @test_throws Lux.SimpleChainsModelConversionError adaptor(ReshapeLayer((2, 3)))

    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        lux_model = Dense(10 => 5)
        adaptor = ToSimpleChainsAdaptor((static(10),))
        smodel = adaptor(lux_model)

        ps, st = Lux.setup(Lux.replicate(rng), smodel) |> dev
        x = randn(rng, 10, 3) |> aType

        if ongpu
            @test_throws ArgumentError smodel(x, ps, st)
        else
            @test size(first(smodel(x, ps, st))) == (5, 3)
        end
    end

    lux_model = Dense(10 => 5)
    adaptor = ToSimpleChainsAdaptor((static(10),))
    smodel = adaptor(lux_model)

    smodel2 = SimpleChainsLayer(smodel.layer, Val(true))
    @test smodel2.layer == smodel.layer
    @test smodel2.lux_layer === nothing
end

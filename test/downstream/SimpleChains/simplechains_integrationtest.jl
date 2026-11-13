using Lux, StableRNGs, Test, LuxTestUtils, Zygote
import SimpleChains: static

@testset "SimpleChains Integration" begin
    lux_model = Chain(
        Conv((5, 5), 1 => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        FlattenLayer(3),
        Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)),
    )

    adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))

    simple_chains_model = adaptor(lux_model)

    rng = StableRNG(0)
    ps_rng = []
    for i in 1:2
        ps, st = Lux.setup(Lux.replicate(rng), simple_chains_model)
        push!(ps_rng, ps)
    end
    @test allequal(ps_rng)

    ps, st = Lux.setup(StableRNG(0), simple_chains_model)

    x = randn(Float32, 28, 28, 1, 1)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 1)

    __f = (x, p) -> sum(first(simple_chains_model(x, p, st)))

    gs = Zygote.gradient(__f, x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    # See https://github.com/LuxDL/Lux.jl/issues/644
    @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3, broken_backends=[AutoEnzyme()])

    x = randn(Float32, 28, 28, 1, 15)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 15)

    __f = (x, p) -> sum(first(simple_chains_model(x, p, st)))

    gs = Zygote.gradient(__f, x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    # See https://github.com/LuxDL/Lux.jl/issues/644
    @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3, broken_backends=[AutoEnzyme()])

    @testset "Array Output" begin
        adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)), true)
        simple_chains_model = adaptor(lux_model)

        ps, st = Lux.setup(StableRNG(0), simple_chains_model)
        x = randn(Float32, 28, 28, 1, 1)

        @test size(first(simple_chains_model(x, ps, st))) == (10, 1)
        @test first(simple_chains_model(x, ps, st)) isa Array
    end

    lux_model = Chain(
        FlattenLayer(3), Dense(784 => 20, tanh), Dropout(0.5), Dense(20 => 10)
    )

    adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))

    simple_chains_model = adaptor(lux_model)

    ps, st = Lux.setup(StableRNG(0), simple_chains_model)

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

            simple_chains_model = @test_warn "The model provided is not a `Chain`. Trying to wrap it into a `Chain` but this might fail. Please consider using `Chain` directly." adaptor(
                lux_model
            )

            ps, st = Lux.setup(StableRNG(0), simple_chains_model)

            x = randn(Float32, 10, 3)
            @test size(first(simple_chains_model(x, ps, st))) == (5, 3)

            __f = (x, p) -> sum(first(simple_chains_model(x, p, st)))

            gs = Zygote.gradient(__f, x, ps)
            @test size(gs[1]) == size(x)
            @test length(gs[2].params) == length(ps.params)

            # See https://github.com/LuxDL/Lux.jl/issues/644
            @test_gradients(
                __f,
                x,
                ps;
                atol=1.0f-3,
                rtol=1.0f-3,
                broken_backends=[AutoEnzyme()],
                soft_fail=[AutoForwardDiff(), AutoFiniteDiff()]
            )
        end
    end

    # Failures
    @test_throws Lux.SimpleChainsModelConversionException adaptor(
        Conv((1, 1), 2 => 3; stride=(5, 5))
    )
    @test_throws Lux.SimpleChainsModelConversionException adaptor(Dropout(0.2f0; dims=1))
    @test_throws Lux.SimpleChainsModelConversionException adaptor(FlattenLayer())
    @test_throws Lux.SimpleChainsModelConversionException adaptor(
        MaxPool((2, 2); stride=(1, 1))
    )
    @test_throws Lux.SimpleChainsModelConversionException adaptor(ReshapeLayer((2, 3)))

    lux_model = Dense(10 => 5)
    adaptor = ToSimpleChainsAdaptor((static(10),))
    smodel = adaptor(lux_model)

    ps, st = Lux.setup(Lux.replicate(rng), smodel)
    x = randn(rng, 10, 3)

    @test size(first(smodel(x, ps, st))) == (5, 3)

    lux_model = Dense(10 => 5)
    adaptor = ToSimpleChainsAdaptor((static(10),))
    smodel = adaptor(lux_model)

    smodel2 = SimpleChainsLayer(smodel.layer, Val(true))
    @test smodel2.layer == smodel.layer
    @test smodel2.lux_layer === nothing
end

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

    x = randn(Float32, 28, 28, 1, 15)
    @test size(first(simple_chains_model(x, ps, st))) == (10, 15)

    __f = (x, p) -> sum(first(simple_chains_model(x, p, st)))

    gs = Zygote.gradient(__f, x, ps)
    @test size(gs[1]) == size(x)
    @test length(gs[2].params) == length(ps.params)

    # The reference gradient of `@test_gradients` comes from FiniteDiff, which needs two
    # forward passes per parameter and input. On the model above that is ~96k passes,
    # which takes hours under `--code-coverage=user`, so check gradients on a small model.
    @testset "Gradients" begin
        small_model = Chain(
            Conv((3, 3), 1 => 2, relu), MaxPool((2, 2)), FlattenLayer(3), Dense(32 => 3)
        )
        small_sc_model = ToSimpleChainsAdaptor((static(10), static(10), static(1)))(
            small_model
        )
        ps_small, st_small = Lux.setup(StableRNG(0), small_sc_model)

        for batch in (1, 4)
            x_small = randn(StableRNG(1), Float32, 10, 10, 1, batch)
            __f_small = (x, p) -> sum(first(small_sc_model(x, p, st_small)))

            # See https://github.com/LuxDL/Lux.jl/issues/644
            @test_gradients(
                __f_small,
                x_small,
                ps_small;
                atol=1.0f-3,
                rtol=1.0f-3,
                broken_backends=[AutoEnzyme(), AutoMooncake()]
            )
        end
    end

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
                broken_backends=[AutoEnzyme(), AutoMooncake()],
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

    # Without the Lux model we fall back to the SimpleChains initialization
    ps2, st2 = Lux.setup(StableRNG(0), smodel2)
    @test length(ps2.params) == Lux.parameterlength(lux_model)
    @test size(first(smodel2(randn(Float32, 10, 3), ps2, st2))) == (5, 3)
end

@testset "Preserve initialization functions: LuxDL/Lux.jl#582" begin
    @testset "Dense" begin
        small_init(rng, dims...) = fill(1.0f-3, dims...)
        lux_model = Chain(
            Dense(4 => 8, tanh),
            Dense(8 => 8, tanh; use_bias=false),
            Dense(8 => 2; init_weight=small_init, init_bias=small_init),
        )
        smodel = ToSimpleChainsAdaptor((static(4),), true)(lux_model)

        ps, st = Lux.setup(StableRNG(0), smodel)
        ps_lux, st_lux = Lux.setup(StableRNG(0), lux_model)

        @test ps.params isa Vector{Float32}
        @test ps.params == vcat(
            vec(ps_lux.layer_1.weight),
            ps_lux.layer_1.bias,
            vec(ps_lux.layer_2.weight),
            vec(ps_lux.layer_3.weight),
            ps_lux.layer_3.bias,
        )
        @test all(==(1.0f-3), ps.params[(end - 2 * 8 - 2 + 1):end])

        x = randn(StableRNG(1), Float32, 4, 3)
        @test first(smodel(x, ps, st)) ≈ first(lux_model(x, ps_lux, st_lux))
    end

    @testset "eltype of the initialization functions is kept" begin
        lux_model = Chain(
            Dense(3 => 4, tanh; init_weight=randn64, init_bias=randn64),
            Dense(4 => 2; init_weight=randn64, init_bias=randn64),
        )
        smodel = ToSimpleChainsAdaptor((static(3),), true)(lux_model)

        ps, st = Lux.setup(StableRNG(0), smodel)
        ps_lux, st_lux = Lux.setup(StableRNG(0), lux_model)

        @test ps.params isa Vector{Float64}
        x = randn(StableRNG(1), Float64, 3, 5)
        @test first(smodel(x, ps, st)) ≈ first(lux_model(x, ps_lux, st_lux))
    end

    @testset "Conv, MaxPool and FlattenLayer" begin
        for cross_correlation in (false, true), use_bias in (false, true)
            lux_model = Chain(
                Conv((3, 3), 2 => 4, relu; cross_correlation, use_bias),
                MaxPool((2, 2)),
                Conv((3, 3), 4 => 3; cross_correlation, use_bias),
                FlattenLayer(3),
                Dense(27 => 2),
            )
            smodel = ToSimpleChainsAdaptor((static(12), static(12), static(2)), true)(
                lux_model
            )

            ps, st = Lux.setup(StableRNG(0), smodel)
            ps_lux, st_lux = Lux.setup(StableRNG(0), lux_model)

            @test ps.params isa Vector{Float32}
            x = randn(StableRNG(1), Float32, 12, 12, 2, 3)
            @test first(smodel(x, ps, st)) ≈ first(lux_model(x, ps_lux, st_lux))
        end
    end

    @testset "Dropout" begin
        lux_model = Chain(Dense(4 => 8, relu), Dropout(0.5), Dense(8 => 2))
        smodel = ToSimpleChainsAdaptor((static(4),), true)(lux_model)

        ps, _ = Lux.setup(StableRNG(0), smodel)
        ps_lux, _ = Lux.setup(StableRNG(0), lux_model)

        @test ps.params == vcat(
            vec(ps_lux.layer_1.weight),
            ps_lux.layer_1.bias,
            vec(ps_lux.layer_3.weight),
            ps_lux.layer_3.bias,
        )
    end
end

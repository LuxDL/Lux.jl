@testitem "Miscellaneous Layers" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Reshape Layer" begin
            layer = ReshapeLayer((2, 3))
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, 6, 3) |> aType

            @test size(layer(x, ps, st)[1]) == (2, 3, 3)
            @test Lux.outputsize(layer, x, rng) == (2, 3)

            @jet layer(x, ps, st)

            __f = x -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "Reverse Sequence" begin
            layer = ReverseSequence(nothing)
            layer2 = ReverseSequence(2)
            layer3 = ReverseSequence(1)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            ps2, st2 = Lux.setup(rng, layer2) |> dev
            ps3, st3 = Lux.setup(rng, layer3) |> dev

            x = randn(rng, 3) |> aType
            xr = reverse(x)
            x2 = randn(rng, 2, 2) |> aType
            x2rd1 = reverse(x2; dims=1)
            x2rd2 = reverse(x2; dims=2)

            xs = randn(rng, 1) |> aType

            @test layer(x, ps, st)[1] == aType(xr)
            @test layer(x2, ps, st)[1] == aType(x2rd1)
            @test_throws ArgumentError layer2(x, ps2, st2)[1]
            @test layer3(x, ps3, st3)[1] == aType(xr)
            @test layer2(x2, ps2, st2)[1] == aType(x2rd2)

            @test layer(xs, ps, st)[1] == aType(xs)

            @jet layer(x, ps, st)

            __f = x -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "Flatten Layer" begin
            layer = FlattenLayer()
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, 6, 3, 2) |> aType

            @test size(layer(x, ps, st)[1]) == (18, 2)

            @jet layer(x, ps, st)

            __f = x -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "NoOpLayer" begin
            layer = NoOpLayer()
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = (x=2, b=5) # Something totally arbitrary

            @test layer(x, ps, st)[1] == x

            @jet layer(x, ps, st)

            x = randn(rng, 6, 3) |> aType
            __f = x -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "SelectDim Layer" begin
            layer = SelectDim(3, 1)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, 6, 4, 3, 2) |> aType

            @test size(layer(x, ps, st)[1]) == (6, 4, 2)

            @jet layer(x, ps, st)

            __f = x -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "WrappedFunction" begin
            layer = WrappedFunction(x -> x .* x)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, 6, 4, 3, 2) |> aType

            @test layer(x, ps, st)[1] == x .* x

            @jet layer(x, ps, st)

            __f = x -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end

@testitem "Dense" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "constructors" begin
            layer = Dense(10, 100)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(ps.weight) == (100, 10)
            @test size(ps.bias) == (100,)
            @test layer.activation == identity

            layer = Dense(10, 100, relu; use_bias=false)
            ps, st = Lux.setup(rng, layer) |> dev

            @test !haskey(ps, :bias)
            @test layer.activation == relu
        end

        @testset "dimensions" begin
            layer = Dense(10, 5)
            ps, st = Lux.setup(rng, layer)

            @test size(first(Lux.apply(layer, randn(10), ps, st))) == (5,)
            @test size(first(Lux.apply(layer, randn(10, 2), ps, st))) == (5, 2)

            @test LuxCore.outputsize(layer, randn(10), rng) == (5,)
        end

        @testset "zeros" begin
            @test begin
                layer = Dense(10, 1, identity; init_weight=ones32, init_bias=zeros32)
                first(Lux.apply(
                    layer, ones(10, 1) |> aType, dev.(Lux.setup(rng, layer))...))
            end == 10 * aType(ones(1, 1))

            @test begin
                layer = Dense(10, 1, identity; init_weight=ones32, init_bias=zeros32)
                first(Lux.apply(
                    layer, ones(10, 2) |> aType, dev.(Lux.setup(rng, layer))...))
            end == 10 * aType(ones(1, 2))

            @test begin
                layer = Dense(10, 2, identity; init_weight=ones32, init_bias=zeros32)
                first(Lux.apply(
                    layer, ones(10, 1) |> aType, dev.(Lux.setup(rng, layer))...))
            end == 10 * aType(ones(2, 1))

            @test begin
                layer = Dense(10, 2, identity; init_weight=ones32, init_bias=zeros32)
                first(Lux.apply(layer, aType([ones(10, 1) 2 * ones(10, 1)]),
                    dev.(Lux.setup(rng, layer))...))
            end == aType([10 20; 10 20])

            @test begin
                layer = Dense(10, 2, identity; init_weight=ones32, use_bias=false)
                first(Lux.apply(layer, aType([ones(10, 1) 2 * ones(10, 1)]),
                    dev.(Lux.setup(rng, layer))...))
            end == aType([10 20; 10 20])
        end
    end
end

@testitem "Dense StaticArrays" setup=[SharedTestSetup] tags=[:core_layers] begin
    using StaticArrays, Enzyme, ForwardDiff, ComponentArrays

    if LuxTestUtils.ENZYME_TESTING_ENABLED
        N = 8
        d = Lux.Dense(N => N)
        ps = (;
            weight=randn(SMatrix{N, N, Float64}),
            bias=randn(SVector{N, Float64})
        )
        x = randn(SVector{N, Float64})

        fun = let d = d, x = x
            ps -> sum(d(x, ps, (;))[1])
        end
        grad1 = ForwardDiff.gradient(fun, ComponentVector(ps))
        grad2 = Enzyme.gradient(Enzyme.Reverse, fun, ps)[1]
        @test maximum(abs, grad1 .- ComponentVector(grad2)) < 1e-6
    end
end

@testitem "Scale" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "constructors" begin
            layer = Scale(10, 100)
            ps, st = Lux.setup(rng, layer) |> dev

            @test Lux.parameterlength(layer) == Lux.parameterlength(ps)
            @test Lux.statelength(layer) == Lux.statelength(st)

            @test size(ps.weight) == (10, 100)
            @test size(ps.bias) == (10, 100)
            @test layer.activation == identity

            layer = Scale(10, 100, relu; use_bias=false)
            ps, st = Lux.setup(rng, layer) |> dev

            @test Lux.parameterlength(layer) == Lux.parameterlength(ps)
            @test Lux.statelength(layer) == Lux.statelength(st)

            @test !haskey(ps, :bias)
            @test layer.activation == relu
        end

        @testset "dimensions" begin
            layer = Scale(10, 5)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(first(Lux.apply(layer, randn(10) |> aType, ps, st))) == (10, 5)
            @test size(first(Lux.apply(layer, randn(10, 5, 2) |> aType, ps, st))) ==
                  (10, 5, 2)

            @test LuxCore.outputsize(layer, randn(10), rng) == (10, 5)
        end

        @testset "zeros" begin
            @test begin
                layer = Scale(10, 1, identity; init_weight=ones32)
                first(Lux.apply(
                    layer, ones(10, 1) |> aType, dev.(Lux.setup(rng, layer))...))
            end == aType(ones(10, 1))

            @test begin
                layer = Scale(10, 1, identity; init_weight=ones32)
                first(Lux.apply(
                    layer, ones(10, 2) |> aType, dev.(Lux.setup(rng, layer))...))
            end == aType(ones(10, 2))

            @test begin
                layer = Scale(2, identity; init_weight=ones32, init_bias=ones32)
                first(Lux.apply(
                    layer, [1 2; 3 4] |> aType, dev.(Lux.setup(rng, layer))...))
            end == aType([2.0 3.0; 4.0 5.0])

            @test begin
                layer = Scale(2, tanh; use_bias=false,
                    init_weight=(rng, args...; kwargs...) -> zeros(args...; kwargs...))
                first(Lux.apply(
                    layer, [1 2; 3 4] |> aType, dev.(Lux.setup(rng, layer))...))
            end == aType(zeros(2, 2))
        end
    end
end

@testitem "Bilinear" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "SkipConnection recombinator" begin
            d = Dense(2 => 2)
            display(d)
            b = Bilinear((2, 2) => 3)
            display(b)
            layer = SkipConnection(d, b)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, Float32, 2, 1) |> aType

            @test size(layer(x, ps, st)[1]) == (3, 1)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=[AutoEnzyme()])

            d = Dense(2 => 2)
            display(d)
            b = Bilinear((2, 2) => 3; use_bias=false)
            display(b)
            layer = SkipConnection(d, b)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, Float32, 2, 1) |> aType

            @test size(layer(x, ps, st)[1]) == (3, 1)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=[AutoEnzyme()])

            d = Dense(2 => 3)
            display(d)
            b = Bilinear((3, 2) => 5)
            display(b)
            layer = SkipConnection(d, b)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, Float32, 2, 7, 11) |> aType

            @test size(layer(x, ps, st)[1]) == (5, 7, 11)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=[AutoEnzyme()])
        end

        @testset "Two-streams zero sum" begin
            x = zeros(Float32, 2, 1) |> aType
            y = zeros(Float32, 1, 1) |> aType
            layer = Bilinear((2, 1) => 3; init_bias=zeros32)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(layer((x, y), ps, st)[1]) == (3, 1)
            @test sum(abs2, layer((x, y), ps, st)[1]) == 0.0f0

            @test LuxCore.outputsize(layer, (x, y), rng) == (3,)

            @jet layer((x, y), ps, st)

            __f = (x, y, ps) -> sum(first(layer((x, y), ps, st)))
            @test_gradients(__f, x, y, ps; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=[AutoEnzyme()])
        end

        @testset "Inner interactions" begin
            x = randn(Float32, 2, 1) |> aType
            layer = Bilinear((2, 2) => 3)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(layer(x, ps, st)[1]) == (3, 1)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=[AutoEnzyme()])

            x = randn(Float32, 2, 1) |> aType
            layer = Bilinear(2 => 3)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(layer(x, ps, st)[1]) == (3, 1)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3,
                skip_backends=[AutoEnzyme()])
        end
    end
end

@testitem "Embedding" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Linear indices" begin
            vocab_size, embed_size = 10, 4
            layer = Embedding(vocab_size => embed_size)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(ps.weight) == (embed_size, vocab_size)

            @test LuxCore.outputsize(layer, nothing, rng) == (4,)

            x = rand(1:vocab_size, 1)[1]
            y, st_ = layer(x, ps, st)
            @test size(layer(x, ps, st)[1]) == (embed_size,)
            @test y == ps.weight[:, x]

            @jet layer(x, ps, st)

            x = rand(1:vocab_size, 3) |> aType
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32}
            @test y == ps.weight[:, x]

            @jet layer(x, ps, st)

            x = rand(1:vocab_size, 3, 4) |> aType
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32, 3}
            @test size(y) == (embed_size, 3, 4)

            @jet layer(x, ps, st)
        end

        @testset "Cartesian indices" begin
            vocab_size, embed_size = (5, 2), 4
            layer = Embedding(vocab_size => embed_size)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(ps.weight) == (embed_size, vocab_size...)

            @test LuxCore.outputsize(layer, nothing, rng) == (4,)

            x = (rand(1:vocab_size[1], 1)[1], rand(1:vocab_size[2], 1)[1])
            y, st_ = layer(x, ps, st)
            @test size(layer(x, ps, st)[1]) == (embed_size,)
            @test y == ps.weight[:, x...]

            @jet layer(x, ps, st)

            x = (rand(1:vocab_size[1], 3), rand(1:vocab_size[2], 3)) .|> aType
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32}
            @test y == ps.weight[:, CartesianIndex.(x...)]

            @jet layer(x, ps, st)

            x = (rand(1:vocab_size[1], 3, 4), rand(1:vocab_size[2], 3, 4)) .|> aType
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32, 3}
            @test size(y) == (embed_size, 3, 4)

            @jet layer(x, ps, st)

            x = (rand(1:vocab_size[1], 3), rand(1:vocab_size[2], 4)) .|> aType
            @test_throws DimensionMismatch layer(x, ps, st)

            x = (rand(1:vocab_size[1], 3, 4), rand(1:vocab_size[2], 4, 5)) .|> aType
            @test_throws DimensionMismatch layer(x, ps, st)

            x = ()
            @test_throws ArgumentError layer(x, ps, st)
        end
    end
end

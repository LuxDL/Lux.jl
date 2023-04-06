using Lux, NNlib, Random, Test

include("../test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "$mode: Miscellaneous Layers" for (mode, aType, device, ongpu) in MODES
    @testset "Reshape Layer" begin
        layer = ReshapeLayer((2, 3))
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device
        x = randn(rng, 6, 3) |> aType

        @test size(layer(x, ps, st)[1]) == (2, 3, 3)

        @jet layer(x, ps, st)
        __f = x -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3
    end

    @testset "Flatten Layer" begin
        layer = FlattenLayer()
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device
        x = randn(rng, 6, 3, 2) |> aType

        @test size(layer(x, ps, st)[1]) == (18, 2)

        @jet layer(x, ps, st)
        __f = x -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3
    end

    @testset "NoOpLayer" begin
        layer = NoOpLayer()
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device
        x = (x=2, b=5) # Something totally arbitrary

        @test layer(x, ps, st)[1] == x

        @jet layer(x, ps, st)

        x = randn(rng, 6, 3) |> aType
        __f = x -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3
    end

    @testset "SelectDim Layer" begin
        layer = SelectDim(3, 1)
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device
        x = randn(rng, 6, 4, 3, 2) |> aType

        @test size(layer(x, ps, st)[1]) == (6, 4, 2)

        @jet layer(x, ps, st)
        __f = x -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3
    end

    @testset "WrappedFunction" begin
        layer = WrappedFunction(x -> x .* x)
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device
        x = randn(rng, 6, 4, 3, 2) |> aType

        @test layer(x, ps, st)[1] == x .* x

        @jet layer(x, ps, st)
        __f = x -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3
    end

    @testset "ActivationFunction" begin
        layer = ActivationFunction(tanh)
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device
        x = randn(rng, 6, 4, 3, 2) |> aType

        @test layer(x, ps, st)[1] == tanh.(x)

        @jet layer(x, ps, st)
        __f = x -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3
    end
end

@testset "$mode: Dense" for (mode, aType, device, ongpu) in MODES
    @testset "constructors" begin
        layer = Dense(10, 100)
        ps, st = Lux.setup(rng, layer) .|> device

        @test size(ps.weight) == (100, 10)
        @test size(ps.bias) == (100, 1)
        @test layer.activation == identity

        layer = Dense(10, 100, relu; use_bias=false)
        ps, st = Lux.setup(rng, layer) .|> device

        @test !haskey(ps, :bias)
        @test layer.activation == relu
    end

    @testset "allow fast activation" begin
        layer = Dense(10, 10, tanh)
        @test layer.activation == tanh_fast
        layer = Dense(10, 10, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
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
            first(Lux.apply(layer, ones(10, 1) |> aType, device.(Lux.setup(rng, layer))...))
        end == 10 * aType(ones(1, 1))

        @test begin
            layer = Dense(10, 1, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 2) |> aType, device.(Lux.setup(rng, layer))...))
        end == 10 * aType(ones(1, 2))

        @test begin
            layer = Dense(10, 2, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 1) |> aType, device.(Lux.setup(rng, layer))...))
        end == 10 * aType(ones(2, 1))

        @test begin
            layer = Dense(10, 2, identity; init_weight=ones)
            first(Lux.apply(layer, aType([ones(10, 1) 2 * ones(10, 1)]),
                            device.(Lux.setup(rng, layer))...))
        end == aType([10 20; 10 20])

        @test begin
            layer = Dense(10, 2, identity; init_weight=ones, use_bias=false)
            first(Lux.apply(layer, aType([ones(10, 1) 2 * ones(10, 1)]),
                            device.(Lux.setup(rng, layer))...))
        end == aType([10 20; 10 20])
    end

    # Deprecated Functionality (Remove in v0.5)
    @testset "Deprecations" begin
        @test_deprecated Dense(10, 100, relu; bias=false)
        @test_deprecated Dense(10, 100, relu; bias=true)
        @test_throws ArgumentError Dense(10, 100, relu; bias=false, use_bias=false)
    end
end

@testset "$mode: Scale" for (mode, aType, device, ongpu) in MODES
    @testset "constructors" begin
        layer = Scale(10, 100)
        ps, st = Lux.setup(rng, layer) .|> device

        @test size(ps.weight) == (10, 100)
        @test size(ps.bias) == (10, 100)
        @test layer.activation == identity

        layer = Scale(10, 100, relu; use_bias=false)
        ps, st = Lux.setup(rng, layer) .|> device

        @test !haskey(ps, :bias)
        @test layer.activation == relu
    end

    @testset "allow fast activation" begin
        layer = Scale(10, 5, tanh)
        @test layer.activation == tanh_fast
        layer = Scale(10, 5, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end

    @testset "dimensions" begin
        layer = Scale(10, 5)
        ps, st = Lux.setup(rng, layer) .|> device

        @test size(first(Lux.apply(layer, randn(10) |> aType, ps, st))) == (10, 5)
        @test size(first(Lux.apply(layer, randn(10, 5, 2) |> aType, ps, st))) == (10, 5, 2)
    end

    @testset "zeros" begin
        @test begin
            layer = Scale(10, 1, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 1) |> aType, device.(Lux.setup(rng, layer))...))
        end == aType(ones(10, 1))

        @test begin
            layer = Scale(10, 1, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 2) |> aType, device.(Lux.setup(rng, layer))...))
        end == aType(ones(10, 2))

        @test begin
            layer = Scale(2, identity; init_weight=ones, init_bias=ones)
            first(Lux.apply(layer, [1 2; 3 4] |> aType, device.(Lux.setup(rng, layer))...))
        end == aType([2.0 3.0; 4.0 5.0])

        @test begin
            layer = Scale(2, tanh; bias=false, init_weight=zeros)
            first(Lux.apply(layer, [1 2; 3 4] |> aType, device.(Lux.setup(rng, layer))...))
        end == aType(zeros(2, 2))
    end

    # Deprecated Functionality (Remove in v0.5)
    @testset "Deprecations" begin
        @test_deprecated Scale(10, 100, relu; bias=false)
        @test_deprecated Scale(10, 100, relu; bias=true)
        @test_throws ArgumentError Scale(10, 100, relu; bias=false, use_bias=false)
    end
end

@testset "$mode: Bilinear" for (mode, aType, device, ongpu) in MODES
    @testset "SkipConnection recombinator" begin
        d = Dense(2 => 2)
        display(d)
        b = Bilinear((2, 2) => 3)
        display(b)
        layer = SkipConnection(d, b)
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device
        x = randn(rng, Float32, 2, 1) |> aType

        @test size(layer(x, ps, st)[1]) == (3, 1)

        @jet layer(x, ps, st)
        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

        d = Dense(2 => 2)
        display(d)
        b = Bilinear((2, 2) => 3; use_bias=false)
        display(b)
        layer = SkipConnection(d, b)
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device
        x = randn(rng, Float32, 2, 1) |> aType

        @test size(layer(x, ps, st)[1]) == (3, 1)

        @jet layer(x, ps, st)
        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
    end

    @testset "Two-streams zero sum" begin
        x = zeros(Float32, 2, 1) |> aType
        y = zeros(Float32, 1, 1) |> aType
        layer = Bilinear((2, 1) => 3)
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device

        @test size(layer((x, y), ps, st)[1]) == (3, 1)
        @test sum(abs2, layer((x, y), ps, st)[1]) == 0.0f0

        @jet layer((x, y), ps, st)
        __f = (x, y, ps) -> sum(first(layer((x, y), ps, st)))
        @eval @test_gradients $__f $x $y $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
    end

    @testset "Inner interactions" begin
        x = randn(Float32, 2, 1) |> aType
        layer = Bilinear((2, 2) => 3)
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device

        @test size(layer(x, ps, st)[1]) == (3, 1)

        @jet layer(x, ps, st)
        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

        x = randn(Float32, 2, 1) |> aType
        layer = Bilinear(2 => 3)
        display(layer)
        ps, st = Lux.setup(rng, layer) .|> device

        @test size(layer(x, ps, st)[1]) == (3, 1)

        @jet layer(x, ps, st)
        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
    end
end

@testset "$mode: Embedding" for (mode, aType, device, ongpu) in MODES
    vocab_size, embed_size = 10, 4
    layer = Embedding(vocab_size => embed_size)
    display(layer)
    ps, st = Lux.setup(rng, layer) .|> device

    @test size(ps.weight) == (embed_size, vocab_size)

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

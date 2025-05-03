@testitem "Miscellaneous Layers" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Reshape Layer" begin
            layer = ReshapeLayer((2, 3))
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(randn(rng, 6, 3))

            @test size(layer(x, ps, st)[1]) == (2, 3, 3)
            @test Lux.outputsize(layer, x, rng) == (2, 3)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "Reverse Sequence" begin
            layer = ReverseSequence(nothing)
            layer2 = ReverseSequence(2)
            layer3 = ReverseSequence(1)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            ps2, st2 = dev(Lux.setup(rng, layer2))
            ps3, st3 = dev(Lux.setup(rng, layer3))

            x = aType(randn(rng, 3))
            xr = reverse(x)
            x2 = aType(randn(rng, 2, 2))
            x2rd1 = reverse(x2; dims=1)
            x2rd2 = reverse(x2; dims=2)

            xs = aType(randn(rng, 1))

            @test layer(x, ps, st)[1] == aType(xr)
            @test layer(x2, ps, st)[1] == aType(x2rd1)
            @test_throws ArgumentError layer2(x, ps2, st2)[1]
            @test layer3(x, ps3, st3)[1] == aType(xr)
            @test layer2(x2, ps2, st2)[1] == aType(x2rd2)

            @test layer(xs, ps, st)[1] == aType(xs)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "Flatten Layer" begin
            layer = FlattenLayer()
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(randn(rng, 6, 3, 2))

            @test size(layer(x, ps, st)[1]) == (18, 2)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "NoOpLayer" begin
            layer = NoOpLayer()
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = (x=2, b=5) # Something totally arbitrary

            @test layer(x, ps, st)[1] == x
            @jet layer(x, ps, st)

            x = aType(randn(rng, 6, 3))
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "SelectDim Layer" begin
            layer = SelectDim(3, 1)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(randn(rng, 6, 4, 3, 2))

            @test size(layer(x, ps, st)[1]) == (6, 4, 2)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            layer = SelectDim(2, 1:2)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(randn(rng, 6, 4, 3, 2))
            @test size(layer(x, ps, st)[1]) == (6, 2, 3, 2)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "WrappedFunction" begin
            layer = WrappedFunction(x -> x .* x)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(randn(rng, 6, 4, 3, 2))

            @test layer(x, ps, st)[1] == x .* x
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end

@testitem "Dense" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "constructors" begin
            layer = Dense(10, 100)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(ps.weight) == (100, 10)
            @test size(ps.bias) == (100,)
            @test layer.activation == identity

            layer = Dense(10, 100, relu; use_bias=false)
            ps, st = dev(Lux.setup(rng, layer))

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
                first(
                    Lux.apply(layer, aType(ones(10, 1)), dev.(Lux.setup(rng, layer))...)
                )
            end == 10 * aType(ones(1, 1))

            @test begin
                layer = Dense(10, 1, identity; init_weight=ones32, init_bias=zeros32)
                first(
                    Lux.apply(layer, aType(ones(10, 2)), dev.(Lux.setup(rng, layer))...)
                )
            end == 10 * aType(ones(1, 2))

            @test begin
                layer = Dense(10, 2, identity; init_weight=ones32, init_bias=zeros32)
                first(
                    Lux.apply(layer, aType(ones(10, 1)), dev.(Lux.setup(rng, layer))...)
                )
            end == 10 * aType(ones(2, 1))

            @test begin
                layer = Dense(10, 2, identity; init_weight=ones32, init_bias=zeros32)
                first(
                    Lux.apply(
                        layer,
                        aType([ones(10, 1) 2 * ones(10, 1)]),
                        dev.(Lux.setup(rng, layer))...,
                    ),
                )
            end == aType([10 20; 10 20])

            @test begin
                layer = Dense(10, 2, identity; init_weight=ones32, use_bias=false)
                first(
                    Lux.apply(
                        layer,
                        aType([ones(10, 1) 2 * ones(10, 1)]),
                        dev.(Lux.setup(rng, layer))...,
                    ),
                )
            end == aType([10 20; 10 20])
        end

        @testset "outputsize" begin
            d = Dense(10, 2)
            @test Lux.outputsize(d, randn(Float32, 10, 3), rng) == (2,)
            @test Lux.outputsize(d, randn(Float32, 10), rng) == (2,)
            @test Lux.outputsize(d, randn(Float32, 10, 3, 2), rng) == (2, 3)
        end
    end
end

@testitem "Dense StaticArrays" setup = [SharedTestSetup] tags = [:core_layers] begin
    using StaticArrays, Enzyme, ForwardDiff, ComponentArrays

    if LuxTestUtils.ENZYME_TESTING_ENABLED
        N = 8
        d = Lux.Dense(N => N)
        ps = (; weight=randn(SMatrix{N,N,Float64}), bias=randn(SVector{N,Float64}))
        x = randn(SVector{N,Float64})

        broken = pkgversion(Enzyme) == v"0.13.18"

        @test begin
            grad1 = ForwardDiff.gradient(ComponentArray(ps)) do ps
                sumabs2first(d, x, ps, (;))
            end

            grad2 = Enzyme.gradient(
                Enzyme.Reverse, sumabs2first, Const(d), Const(x), ps, Const((;))
            )[3]

            maximum(abs, grad1 .- ComponentArray(grad2)) < 1.0e-6
        end broken = broken
    end
end

@testitem "Scale" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "constructors" begin
            layer = Scale(10, 100)
            ps, st = dev(Lux.setup(rng, layer))

            @test Lux.parameterlength(layer) == Lux.parameterlength(ps)
            @test Lux.statelength(layer) == Lux.statelength(st)

            @test size(ps.weight) == (10, 100)
            @test size(ps.bias) == (10, 100)
            @test layer.activation == identity

            layer = Scale(10, 100, relu; use_bias=false)
            ps, st = dev(Lux.setup(rng, layer))

            @test Lux.parameterlength(layer) == Lux.parameterlength(ps)
            @test Lux.statelength(layer) == Lux.statelength(st)

            @test !haskey(ps, :bias)
            @test layer.activation == relu
        end

        @testset "dimensions" begin
            layer = Scale(10, 5)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(first(Lux.apply(layer, aType(randn(10)), ps, st))) == (10, 5)
            @test size(first(Lux.apply(layer, aType(randn(10, 5, 2)), ps, st))) ==
                (10, 5, 2)

            @test LuxCore.outputsize(layer, randn(10), rng) == (10, 5)
        end

        @testset "zeros" begin
            @test begin
                layer = Scale(10, 1, identity; init_weight=ones32)
                first(
                    Lux.apply(layer, aType(ones(10, 1)), dev.(Lux.setup(rng, layer))...)
                )
            end == aType(ones(10, 1))

            @test begin
                layer = Scale(10, 1, identity; init_weight=ones32)
                first(
                    Lux.apply(layer, aType(ones(10, 2)), dev.(Lux.setup(rng, layer))...)
                )
            end == aType(ones(10, 2))

            @test begin
                layer = Scale(2, identity; init_weight=ones32, init_bias=ones32)
                first(
                    Lux.apply(layer, aType([1 2; 3 4]), dev.(Lux.setup(rng, layer))...)
                )
            end == aType([2.0 3.0; 4.0 5.0])

            @test begin
                layer = Scale(
                    2,
                    tanh;
                    use_bias=false,
                    init_weight=(rng, args...; kwargs...) -> zeros(args...; kwargs...),
                )
                first(
                    Lux.apply(layer, aType([1 2; 3 4]), dev.(Lux.setup(rng, layer))...)
                )
            end == aType(zeros(2, 2))
        end
    end
end

@testitem "Bilinear" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    skip_backends = VERSION < v"1.11-" ? [AutoEnzyme()] : []

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "SkipConnection recombinator" begin
            d = Dense(2 => 2)
            display(d)
            b = Bilinear((2, 2) => 3)
            display(b)
            layer = SkipConnection(d, b)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(randn(rng, Float32, 2, 1))

            @test size(layer(x, ps, st)[1]) == (3, 1)
            @jet layer(x, ps, st)
            @test_gradients(
                sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )

            d = Dense(2 => 2)
            display(d)
            b = Bilinear((2, 2) => 3; use_bias=false)
            display(b)
            layer = SkipConnection(d, b)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(randn(rng, Float32, 2, 1))

            @test size(layer(x, ps, st)[1]) == (3, 1)
            @jet layer(x, ps, st)
            @test_gradients(
                sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )

            d = Dense(2 => 3)
            display(d)
            b = Bilinear((3, 2) => 5)
            display(b)
            layer = SkipConnection(d, b)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(randn(rng, Float32, 2, 7, 11))

            @test size(layer(x, ps, st)[1]) == (5, 7, 11)
            @jet layer(x, ps, st)
            @test_gradients(
                sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )
        end

        @testset "Two-streams zero sum" begin
            x = aType(zeros(Float32, 2, 1))
            y = aType(zeros(Float32, 1, 1))
            layer = Bilinear((2, 1) => 3; init_bias=zeros32)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(layer((x, y), ps, st)[1]) == (3, 1)
            @test sum(abs2, layer((x, y), ps, st)[1]) == 0.0f0

            @test LuxCore.outputsize(layer, (x, y), rng) == (3,)
            @jet layer((x, y), ps, st)
            @test_gradients(
                sumabs2first, layer, (x, y), ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )
        end

        @testset "Inner interactions" begin
            x = aType(randn(Float32, 2, 1))
            layer = Bilinear((2, 2) => 3)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(layer(x, ps, st)[1]) == (3, 1)
            @jet layer(x, ps, st)
            @test_gradients(
                sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )

            x = aType(randn(Float32, 2, 1))
            layer = Bilinear(2 => 3)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(layer(x, ps, st)[1]) == (3, 1)
            @jet layer(x, ps, st)
            @test_gradients(
                sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )
        end
    end
end

@testitem "Dense cuBLASLt failures with views #1298" setup = [SharedTestSetup] tags = [
    :core_layers
] begin
    using LinearAlgebra

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        mode == "cuda" || continue

        model = Dense(1 => 1, relu; use_bias=false)
        ps, st = dev(Lux.setup(rng, model))
        x = ones(Float32, 1, 2) |> aType
        x = selectdim(x, 2, 1:1) |> transpose

        @test x isa LinearAlgebra.Transpose
        @test size(model(x, ps, st)[1]) == (1, 1)
    end
end

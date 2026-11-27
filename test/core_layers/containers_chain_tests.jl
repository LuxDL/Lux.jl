using ComponentArrays

include("../shared_testsetup.jl")

@testset "Chain" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        layer = Chain(Dense(10 => 5, sigmoid), Dense(5 => 2, tanh), Dense(2 => 1))
        display(layer)
        ps, st = dev(Lux.setup(rng, layer))
        x = aType(rand(Float32, 10, 1))
        y, _ = layer(x, ps, st)
        @test size(y) == (1, 1)
        @test Lux.outputsize(layer, x, rng) == (1,)

        @jet layer(x, ps, st)
        @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

        layer = Chain(;
            l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1)
        )
        display(layer)
        ps, st = dev(Lux.setup(rng, layer))
        x = aType(rand(Float32, 10, 1))
        y, _ = layer(x, ps, st)
        @test size(y) == (1, 1)

        @jet layer(x, ps, st)
        @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

        layer = Chain(;
            l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1)
        )
        display(layer)
        layer = layer[1:2]
        ps, st = dev(Lux.setup(rng, layer))
        x = aType(rand(Float32, 10, 1))
        y, _ = layer(x, ps, st)
        @test size(y) == (2, 1)
        @test Lux.outputsize(layer, x, rng) == (2,)

        @jet layer(x, ps, st)
        @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

        layer = Chain(;
            l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1)
        )
        display(layer)
        layer = layer[begin:(end - 1)]
        ps, st = dev(Lux.setup(rng, layer))
        x = aType(rand(Float32, 10, 1))
        y, _ = layer(x, ps, st)
        @test size(y) == (2, 1)
        @test Lux.outputsize(layer, x, rng) == (2,)

        @jet layer(x, ps, st)
        @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

        layer = Chain(;
            l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1)
        )
        display(layer)
        layer = layer[1]
        ps, st = dev(Lux.setup(rng, layer))
        x = aType(rand(Float32, 10, 1))
        y, _ = layer(x, ps, st)
        @test size(y) == (5, 1)
        @test Lux.outputsize(layer, x, rng) == (5,)

        @jet layer(x, ps, st)
        @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

        @testset "indexing and field access" begin
            encoder = Chain(Dense(10 => 5, sigmoid), Dense(5 => 2, tanh))
            decoder = Chain(Dense(2 => 5, tanh), Dense(5 => 10, sigmoid))
            autoencoder = Chain(; encoder, decoder)
            @test encoder[1] == encoder.layer_1
            @test encoder[2] == encoder.layer_2
            @test autoencoder[1] == autoencoder.encoder
            @test autoencoder[2] == autoencoder.decoder
            @test autoencoder.layers isa NamedTuple
            @test autoencoder.encoder isa Chain
            @test_throws ArgumentError autoencoder.layer_1
        end
    end

    @testset "constructors" begin
        f1(x) = x .+ 1
        f2(x) = x .+ 2
        model = Chain((Dense(2 => 3), Dense(3 => 2)), f1, f2, NoOpLayer())

        @test length(model) == 5

        x = rand(Float32, 2, 5)
        ps, st = Lux.setup(rng, model)

        y = first(model(x, ps, st))
        @test size(y) == (2, 5)

        @test y ≈
            ps.layer_2.weight * (ps.layer_1.weight * x .+ ps.layer_1.bias) .+
              ps.layer_2.bias .+ 1 .+ 2
    end
end

@testset "Maxout" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "constructor" begin
            layer = Maxout(() -> NoOpLayer(), 4)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(rand(rng, Float32, 10, 1))

            @test layer(x, ps, st)[1] == x
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "simple alternatives" begin
            layer1 = Maxout(
                NoOpLayer(), WrappedFunction(x -> 2x), WrappedFunction(x -> 0.5x)
            )
            layer2 = Maxout(;
                l1=NoOpLayer(), l2=WrappedFunction(x -> 2x), l3=WrappedFunction(x -> 0.5x)
            )

            for layer in (layer1, layer2)
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))
                x = aType(Float32.(collect(1:40)))

                @test layer(x, ps, st)[1] == 2 .* x
                @jet layer(x, ps, st)
                @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
            end
        end

        @testset "complex alternatives" begin
            struct MulLayer <: Lux.AbstractLuxLayer end

            (::MulLayer)(x, ps, st) = (ps.weight * x, st)

            A = aType([0.5 0.1]')
            B = aType([0.2 0.7]')
            layer = Maxout(MulLayer(), MulLayer())
            display(layer)

            ps = dev((; layer_1=(; weight=A), layer_2=(; weight=B)))
            st = dev((; layer_1=NamedTuple(), layer_2=NamedTuple()))

            x = aType([3.0 2.0])
            y = aType([0.5, 0.7]) .* x

            @test layer(x, ps, st)[1] == y
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "params" begin
            layer = Maxout(() -> Dense(2, 4), 4)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType([10.0f0 3.0f0]')

            @test Lux.parameterlength(layer) ==
                sum(Lux.parameterlength.(values(layer.layers)))
            @test size(layer(x, ps, st)[1]) == (4, 1)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end

@testset "Repeated" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        LAYERS = [
            Dense(2 => 2),
            Parallel(+, Dense(2 => 2), Dense(2 => 2)),
            Dense(2 => 2),
            Parallel(+, Dense(2 => 2), Dense(2 => 2)),
        ]
        REPEATS = [Val(4), Val(4), Val(4), Val(4)]
        INJECTION = [Val(false), Val(true), Val(false), Val(true)]

        @testset "repeats = $(repeats); input_injection = $(input_injection)" for (
            sublayer, repeats, input_injection
        ) in zip(
            LAYERS, REPEATS, INJECTION
        )
            layer = RepeatedLayer(sublayer; repeats, input_injection)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(rand(rng, Float32, 2, 12))

            y, st_out = layer(x, ps, st)
            @test size(y) == (2, 12)

            res = x
            st_ref = st
            if input_injection == Val(false)
                for _ in 1:4
                    res, st_ref = sublayer(res, ps, st_ref)
                end
            else
                for _ in 1:4
                    res, st_ref = sublayer((res, x), ps, st_ref)
                end
            end
            @test y ≈ res
            @test st_out == st_ref

            @jet layer(x, ps, st) opt_broken = (
                mode == "cuda" && input_injection === Val(true) && VERSION >= v"1.13-"
            )
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "repeats = Val(0) & invalid repeats" begin
            sublayer = Dense(2 => 2)
            ps, st = dev(Lux.setup(rng, sublayer))
            x = aType(rand(rng, Float32, 2, 12))

            for injection in (Val(false), Val(true))
                layer0 = RepeatedLayer(sublayer; repeats=Val(0), input_injection=injection)
                y0, st0 = layer0(x, ps, st)
                @test y0 == x
                @test st0 == st
            end

            @test_throws ArgumentError RepeatedLayer(sublayer; repeats=Val(-1))(x, ps, st)
        end
    end
end

@testset "Coupling Type Instability: #416" begin
    rng = Random.default_rng()

    froggie = Chain(
        BranchLayer(NoOpLayer(), NoOpLayer()),
        Parallel(nothing, Parallel(+, Dense(1 => 1), NoOpLayer()), Dense(1 => 1)),
    )

    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        x = aType([1.0])
        ps, st = Lux.setup(rng, froggie)
        st = dev(st)

        ps_nt = dev(ps)
        @test @inferred(froggie(x, ps_nt, st)) isa Any

        ps_ca = dev(ComponentArray(ps))
        @test @inferred(froggie(x, ps_ca, st)) isa Any broken = ongpu
    end
end

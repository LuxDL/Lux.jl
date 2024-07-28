
@testitem "SkipConnection" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "zero sum" begin
            layer = SkipConnection(
                WrappedFunction{:direct_call}(Broadcast.BroadcastFunction(zero)), .+)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, Float32, 10, 10, 10, 10) |> aType

            @test layer(x, ps, st)[1] == x

            @jet layer(x, ps, st)

            __f = x -> sum(first(layer(x, ps, st)))
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "concat size" begin
            layer = SkipConnection(Dense(10, 10), (a, b) -> hcat(a, b))
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, Float32, 10, 2) |> aType

            @test size(layer(x, ps, st)[1]) == (10, 4)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            # Method ambiguity for concatenation
            test_gradients(
                __f, x, ps; atol=1.0f-3, rtol=1.0f-3, broken_backends=[AutoReverseDiff()])
        end
    end
end

@testitem "Parallel" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "zero sum" begin
            layer = Parallel(
                +, WrappedFunction{:direct_call}(Broadcast.BroadcastFunction(zero)),
                NoOpLayer())
            @test :layer_1 in keys(layer) && :layer_2 in keys(layer)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, 10, 10, 10, 10) |> aType

            @test layer(x, ps, st)[1] == x

            @jet layer(x, ps, st)

            __f = x -> sum(first(layer(x, ps, st)))
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "concat size" begin
            layer = Parallel((a, b) -> cat(a, b; dims=2), Dense(10, 10), NoOpLayer())
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = randn(rng, 10, 2) |> aType

            @test size(layer(x, ps, st)[1]) == (10, 4)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)

            layer = Parallel(hcat, Dense(10, 10), NoOpLayer())
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(layer(x, ps, st)[1]) == (10, 4)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            # Method ambiguity for concatenation
            test_gradients(
                __f, x, ps; atol=1.0f-3, rtol=1.0f-3, broken_backends=[AutoReverseDiff()])
        end

        @testset "vararg input" begin
            layer = Parallel(+, Dense(10, 2), Dense(5, 2), Dense(4, 2))
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = (randn(rng, 10, 1), randn(rng, 5, 1), randn(rng, 4, 1)) .|> aType

            @test size(layer(x, ps, st)[1]) == (2, 1)

            @jet layer(x, ps, st)

            __f = (x1, x2, x3, ps) -> sum(first(layer((x1, x2, x3), ps, st)))
            test_gradients(__f, x[1], x[2], x[3], ps; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "named layers" begin
            layer = Parallel(+; d102=Dense(10, 2), d52=Dense(5, 2), d42=Dense(4, 2))
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = (randn(rng, 10, 1), randn(rng, 5, 1), randn(rng, 4, 1)) .|> aType

            @test size(layer(x, ps, st)[1]) == (2, 1)

            @jet layer(x, ps, st)

            __f = (x1, x2, x3, ps) -> sum(first(layer((x1, x2, x3), ps, st)))
            test_gradients(__f, x[1], x[2], x[3], ps; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "connection is called once" begin
            CNT = Ref(0)
            f_cnt = (x...) -> (CNT[] += 1; +(x...))
            layer = Parallel(
                f_cnt, WrappedFunction(sin), WrappedFunction(cos), WrappedFunction(tan))
            ps, st = Lux.setup(rng, layer) |> dev
            Lux.apply(layer, 1, ps, st)
            @test CNT[] == 1
            @jet layer(1, ps, st)
            Lux.apply(layer, (1, 2, 3), ps, st)
            @test CNT[] == 2
            layer = Parallel(f_cnt, WrappedFunction(sin))
            Lux.apply(layer, 1, ps, st)
            @test CNT[] == 3
        end

        # Ref https://github.com/FluxML/Flux.jl/issues/1673
        @testset "Input domain" begin
            struct Input
                x
            end

            struct L1 <: Lux.AbstractExplicitLayer end
            (::L1)(x, ps, st) = (ps.x * x, st)
            Lux.initialparameters(rng::AbstractRNG, ::L1) = (x=randn(rng, Float32, 3, 3),)
            Base.:*(a::AbstractArray, b::Input) = a * b.x

            par = Parallel(+, L1(), L1())
            ps, st = Lux.setup(rng, par) |> dev

            ip = Input(rand(Float32, 3, 3) |> aType)
            ip2 = Input(rand(Float32, 3, 3) |> aType)

            @test check_approx(par(ip, ps, st)[1],
                par.layers[1](ip.x, ps.layer_1, st.layer_1)[1] +
                par.layers[2](ip.x, ps.layer_2, st.layer_2)[1])
            @test check_approx(par((ip, ip2), ps, st)[1],
                par.layers[1](ip.x, ps.layer_1, st.layer_1)[1] +
                par.layers[2](ip2.x, ps.layer_2, st.layer_2)[1])
            gs = Zygote.gradient((p, x...) -> sum(par(x, p, st)[1]), ps, ip, ip2)
            gs_reg = Zygote.gradient(ps, ip, ip2) do p, x, y
                return sum(par.layers[1](x.x, p.layer_1, st.layer_1)[1] +
                           par.layers[2](y.x, p.layer_2, st.layer_2)[1])
            end

            @test check_approx(gs[1], gs_reg[1])
            @test check_approx(gs[2].x, gs_reg[2].x)
            @test check_approx(gs[3].x, gs_reg[3].x)
        end
    end
end

@testitem "PairwiseFusion" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = (rand(Float32, 1, 10), rand(Float32, 30, 10), rand(Float32, 10, 10)) .|> aType
        layer = PairwiseFusion(+, Dense(1, 30), Dense(30, 10))
        display(layer)
        ps, st = Lux.setup(rng, layer) |> dev
        y, _ = layer(x, ps, st)
        @test size(y) == (10, 10)

        @jet layer(x, ps, st)
        __f = (x1, x2, x3, ps) -> sum(first(layer((x1, x2, x3), ps, st)))
        test_gradients(__f, x[1], x[2], x[3], ps; atol=1.0f-3, rtol=1.0f-3)

        layer = PairwiseFusion(+; d1=Dense(1, 30), d2=Dense(30, 10))
        display(layer)
        ps, st = Lux.setup(rng, layer) |> dev
        y, _ = layer(x, ps, st)
        @test size(y) == (10, 10)
        @jet layer(x, ps, st)

        __f = (x1, x2, x3, ps) -> sum(first(layer((x1, x2, x3), ps, st)))
        test_gradients(__f, x[1], x[2], x[3], ps; atol=1.0f-3, rtol=1.0f-3)

        x = rand(1, 10)
        layer = PairwiseFusion(.+, Dense(1, 10), Dense(10, 1))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        y, _ = layer(x, ps, st)
        @test size(y) == (1, 10)

        @jet layer(x, ps, st)

        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)

        layer = PairwiseFusion(vcat, WrappedFunction(x -> x .+ 1),
            WrappedFunction(x -> x .+ 2), WrappedFunction(x -> x .^ 3))
        display(layer)
        ps, st = Lux.setup(rng, layer) |> dev
        @test layer((2, 10, 20, 40), ps, st)[1] == [125, 1728, 8000, 40]

        layer = PairwiseFusion(vcat, WrappedFunction(x -> x .+ 1),
            WrappedFunction(x -> x .+ 2), WrappedFunction(x -> x .^ 3))
        display(layer)
        ps, st = Lux.setup(rng, layer) |> dev
        @test layer(7, ps, st)[1] == [1000, 729, 343, 7]
    end
end

@testitem "BranchLayer" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        layer = BranchLayer(Dense(10, 10), Dense(10, 10))
        display(layer)
        ps, st = Lux.setup(rng, layer) |> dev
        x = rand(Float32, 10, 1) |> aType
        (y1, y2), _ = layer(x, ps, st)
        @test size(y1) == (10, 1)
        @test size(y2) == (10, 1)
        @test y1 == layer.layers.layer_1(x, ps.layer_1, st.layer_1)[1]
        @test y2 == layer.layers.layer_2(x, ps.layer_2, st.layer_2)[1]

        @jet layer(x, ps, st)

        __f = (x, ps) -> sum(sum, first(layer(x, ps, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)

        layer = BranchLayer(; d1=Dense(10, 10), d2=Dense(10, 10))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = rand(Float32, 10, 1)
        (y1, y2), _ = layer(x, ps, st)
        @test size(y1) == (10, 1)
        @test size(y2) == (10, 1)
        @test y1 == layer.layers.d1(x, ps.d1, st.d1)[1]
        @test y2 == layer.layers.d2(x, ps.d2, st.d2)[1]

        @jet layer(x, ps, st)

        __f = (x, ps) -> sum(sum, first(layer(x, ps, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)
    end
end

@testitem "Chain" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        layer = Chain(Dense(10 => 5, sigmoid), Dense(5 => 2, tanh), Dense(2 => 1))
        display(layer)
        ps, st = Lux.setup(rng, layer) |> dev
        x = rand(Float32, 10, 1) |> aType
        y, _ = layer(x, ps, st)
        @test size(y) == (1, 1)
        @test Lux.outputsize(layer) == (1,)

        @jet layer(x, ps, st)

        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)

        layer = Chain(;
            l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1))
        display(layer)
        ps, st = Lux.setup(rng, layer) |> dev
        x = rand(Float32, 10, 1) |> aType
        y, _ = layer(x, ps, st)
        @test size(y) == (1, 1)

        @jet layer(x, ps, st)

        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)

        layer = Chain(;
            l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1))
        display(layer)
        layer = layer[1:2]
        ps, st = Lux.setup(rng, layer) |> dev
        x = rand(Float32, 10, 1) |> aType
        y, _ = layer(x, ps, st)
        @test size(y) == (2, 1)
        @test Lux.outputsize(layer) == (2,)

        @jet layer(x, ps, st)

        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)

        layer = Chain(;
            l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1))
        display(layer)
        layer = layer[begin:(end - 1)]
        ps, st = Lux.setup(rng, layer) |> dev
        x = rand(Float32, 10, 1) |> aType
        y, _ = layer(x, ps, st)
        @test size(y) == (2, 1)
        @test Lux.outputsize(layer) == (2,)

        @jet layer(x, ps, st)

        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)

        layer = Chain(;
            l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1))
        display(layer)
        layer = layer[1]
        ps, st = Lux.setup(rng, layer) |> dev
        x = rand(Float32, 10, 1) |> aType
        y, _ = layer(x, ps, st)
        @test size(y) == (5, 1)
        @test Lux.outputsize(layer) == (5,)

        @jet layer(x, ps, st)

        __f = (x, ps) -> sum(first(layer(x, ps, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)

        @test_throws ArgumentError Chain(;
            l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh),
            d21=Dense(2 => 1), d2=Dense(2 => 1), disable_optimizations=false)

        @testset "indexing and field access" begin
            encoder = Chain(Dense(10 => 5, sigmoid), Dense(5 => 2, tanh))
            decoder = Chain(Dense(2 => 5, tanh), Dense(5 => 10, sigmoid))
            autoencoder = Chain(; encoder, decoder)
            @test encoder[1] == encoder.layer_1
            @test encoder[2] == encoder.layer_2
            @test autoencoder[1] == autoencoder.encoder
            @test autoencoder[2] == autoencoder.decoder
            @test keys(encoder) == (:layer_1, :layer_2)
            @test keys(autoencoder) == (:encoder, :decoder)
            @test autoencoder.layers isa NamedTuple
            @test autoencoder.encoder isa Chain
            @test_throws ArgumentError autoencoder.layer_1
        end
    end

    @testset "constructors" begin
        @test Chain([Dense(10 => 5, sigmoid)]) == Dense(10 => 5, sigmoid)

        f1(x, ps, st::NamedTuple) = (x .+ 1, st)
        f2(x) = x .+ 2
        model = Chain((Dense(2 => 3), Dense(3 => 2)), f1, f2, NoOpLayer())

        @test length(model) == 4

        x = rand(Float32, 2, 5)
        ps, st = Lux.setup(rng, model)

        y = first(model(x, ps, st))
        @test size(y) == (2, 5)

        @test y ≈
              ps.layer_2.weight * (ps.layer_1.weight * x .+ ps.layer_1.bias) .+
              ps.layer_2.bias .+ 1 .+ 2
    end
end

@testitem "Maxout" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "constructor" begin
            layer = Maxout(() -> NoOpLayer(), 4)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = rand(rng, Float32, 10, 1) |> aType

            @test layer(x, ps, st)[1] == x

            @jet layer(x, ps, st)
        end

        @testset "simple alternatives" begin
            layer1 = Maxout(
                NoOpLayer(), WrappedFunction(x -> 2x), WrappedFunction(x -> 0.5x))
            layer2 = Maxout(;
                l1=NoOpLayer(), l2=WrappedFunction(x -> 2x), l3=WrappedFunction(x -> 0.5x))

            for layer in (layer1, layer2)
                display(layer)
                ps, st = Lux.setup(rng, layer) |> dev
                x = Float32.(collect(1:40)) |> aType

                @test layer(x, ps, st)[1] == 2 .* x

                @jet layer(x, ps, st)

                __f = x -> sum(first(layer(x, ps, st)))
                test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
            end
        end

        @testset "complex alternatives" begin
            A = aType([0.5 0.1]')
            B = aType([0.2 0.7]')
            layer = Maxout(WrappedFunction(x -> A * x), WrappedFunction(x -> B * x))
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = [3.0 2.0] |> aType
            y = aType([0.5, 0.7]) .* x

            @test layer(x, ps, st)[1] == y

            @jet layer(x, ps, st)

            __f = x -> sum(first(layer(x, ps, st)))
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "params" begin
            layer = Maxout(() -> Dense(2, 4), 4)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = [10.0f0 3.0f0]' |> aType

            @test Lux.parameterlength(layer) ==
                  sum(Lux.parameterlength.(values(layer.layers)))
            @test size(layer(x, ps, st)[1]) == (4, 1)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end

@testitem "Repeated" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        LAYERS = [Dense(2 => 2), Parallel(+, Dense(2 => 2), Dense(2 => 2)),
            Dense(2 => 2), Parallel(+, Dense(2 => 2), Dense(2 => 2))]
        REPEATS = [Val(4), Val(4), Val(4), Val(4)]
        INJECTION = [Val(false), Val(true), Val(false), Val(true)]

        @testset "repeats = $(repeats); input_injection = $(input_injection)" for (layer, repeats, input_injection) in zip(
            LAYERS, REPEATS, INJECTION)
            layer = RepeatedLayer(layer; repeats, input_injection)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev
            x = rand(rng, Float32, 2, 12) |> aType

            @test size(layer(x, ps, st)[1]) == (2, 12)

            @jet layer(x, ps, st)

            __f = (x, ps) -> sum(first(layer(x, ps, st)))
            test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end

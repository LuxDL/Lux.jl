using JET, Lux, NNlib, Random

include("../utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "Miscellaneous Layers" begin
    @testset "Reshape Layer" begin
        layer = ReshapeLayer((2, 3))
        println(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 3)

        @test size(layer(x, ps, st)[1]) == (2, 3, 3)
        @test_call layer(x, ps, st)
        @test_opt target_modules = (Lux,)  layer(x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1f-3, rtol=1f-3)
    end

    @testset "Flatten Layer" begin
        layer = FlattenLayer()
        println(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 3, 2)

        @test size(layer(x, ps, st)[1]) == (18, 2)
        @test_call layer(x, ps, st)
        @test_opt target_modules = (Lux,)  layer(x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1f-3, rtol=1f-3)
    end

    @testset "NoOpLayer" begin
        layer = NoOpLayer()
        println(layer)
        ps, st = Lux.setup(rng, layer)
        x = (x = 2, b = 5) # Something totally arbitrary

        @test layer(x, ps, st)[1] == x
        @test_call layer(x, ps, st)
        @test_opt target_modules = (Lux,)  layer(x, ps, st)

        x = randn(rng, 6, 3)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1f-3, rtol=1f-3)
    end

    @testset "SelectDim Layer" begin
        layer = SelectDim(3, 1)
        println(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 4, 3, 2)

        @test size(layer(x, ps, st)[1]) == (6, 4, 2)
        @test_call layer(x, ps, st)
        @test_opt target_modules = (Lux,) layer(x, ps, st) broken=true
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1f-3, rtol=1f-3)
    end

    @testset "WrappedFunction" begin
        layer = WrappedFunction(x -> x .* x)
        println(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 4, 3, 2)

        @test layer(x, ps, st)[1] == x .* x
        @test_call layer(x, ps, st)
        @test_opt target_modules = (Lux,) layer(x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1f-3, rtol=1f-3)
    end

    @testset "ActivationFunction" begin
        layer = ActivationFunction(tanh)
        println(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 6, 4, 3, 2)

        @test layer(x, ps, st)[1] == tanh.(x)
        @test_call layer(x, ps, st)
        @test_opt target_modules = (Lux,) layer(x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1f-3, rtol=1f-3)
    end
end

@testset "Containers" begin
    @testset "SkipConnection" begin
        @testset "zero sum" begin
            layer = SkipConnection(WrappedFunction(zero), (a, b) -> a .+ b)
            println(layer)
            ps, st = Lux.setup(rng, layer)
            x = randn(rng, 10, 10, 10, 10)

            @test layer(x, ps, st)[1] == x
            @test_call layer(x, ps, st)
            @test_opt target_modules = (Lux,) layer(x, ps, st)
            test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1f-3, rtol=1f-3)
        end

        @testset "concat size" begin
            layer = SkipConnection(Dense(10, 10), (a, b) -> hcat(a, b))
            println(layer)
            ps, st = Lux.setup(rng, layer)
            x = randn(rng, 10, 2)

            @test size(layer(x, ps, st)[1]) == (10, 4)
            @test_call layer(x, ps, st)
            @test_opt target_modules = (Lux,) layer(x, ps, st)
            test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1f-3, rtol=1f-3)
        end
    end
end

@testset "Dense" begin
    @testset "constructors" begin
        layer = Dense(10, 100)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (100, 10)
        @test size(ps.bias) == (100, 1)
        @test layer.activation == identity

        layer = Dense(10, 100, relu; bias=false)
        ps, st = Lux.setup(rng, layer)

        @test !haskey(ps, :bias)
        @test layer.activation == relu
    end

    @testset "dimensions" begin
        layer = Dense(10, 5)
        ps, st = Lux.setup(rng, layer)

        # For now only support 2D/1D input
        # @test length(first(Lux.apply(layer, randn(10), ps, st))) == 5
        # @test_throws DimensionMismatch first(Lux.apply(layer, randn(1), ps, st))
        # @test_throws MethodError first(Lux.apply(layer, 1, ps, st))
        @test size(first(Lux.apply(layer, randn(10), ps, st))) == (5,)
        @test size(first(Lux.apply(layer, randn(10, 2), ps, st))) == (5, 2)
        # @test size(first(Lux.apply(layer, randn(10, 2, 3), ps, st))) == (5, 2, 3)
        # @test size(first(Lux.apply(layer, randn(10, 2, 3, 4), ps, st))) == (5, 2, 3, 4)
        # @test_throws DimensionMismatch first(Lux.apply(layer, randn(11, 2, 3), ps, st))
    end

    @testset "zeros" begin
        @test begin
            layer = Dense(10, 1, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 1), Lux.setup(rng, layer)...))
        end == 10 * ones(1, 1)

        @test begin
            layer = Dense(10, 1, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 2), Lux.setup(rng, layer)...))
        end == 10 * ones(1, 2)

        @test begin
            layer = Dense(10, 2, identity; init_weight=ones)
            first(Lux.apply(layer, ones(10, 1), Lux.setup(rng, layer)...))
        end == 10 * ones(2, 1)

        @test begin
            layer = Dense(10, 2, identity; init_weight=ones)
            first(Lux.apply(layer, [ones(10, 1) 2 * ones(10, 1)], Lux.setup(rng, layer)...))
        end == [10 20; 10 20]

        @test begin
            layer = Dense(10, 2, identity; init_weight=ones, bias=false)
            first(Lux.apply(layer, [ones(10, 1) 2 * ones(10, 1)], Lux.setup(rng, layer)...))
        end == [10 20; 10 20]
    end
end

@testset "Parallel" begin
    @testset "zero sum" begin
        input = randn(10, 10, 10, 10)
        layer = Parallel(+, WrappedFunction(zero), NoOpLayer())
        @test layer(input, Lux.setup(rng, layer)...)[1] == input
    end

    @testset "concat size" begin
        input = randn(10, 2)
        layer = Parallel((a, b) -> cat(a, b; dims=2), Dense(10, 10), NoOpLayer())
        @test size(Lux.apply(layer, input, Lux.setup(rng, layer)...)[1]) == (10, 4)
        layer = Parallel(hcat, Dense(10, 10), NoOpLayer())
        @test size(Lux.apply(layer, input, Lux.setup(rng, layer)...)[1]) == (10, 4)
    end

    @testset "vararg input" begin
        inputs = randn(10), randn(5), randn(4)
        layer = Parallel(+, Dense(10, 2), Dense(5, 2), Dense(4, 2))
        @test size(Lux.apply(layer, inputs, Lux.setup(rng, layer)...)[1]) == (2,)
    end

    @testset "connection is called once" begin
        CNT = Ref(0)
        f_cnt = (x...) -> (CNT[] += 1; +(x...))
        layer = Parallel(f_cnt, WrappedFunction(sin), WrappedFunction(cos), WrappedFunction(tan))
        Lux.apply(layer, 1, Lux.setup(rng, layer)...)
        @test CNT[] == 1
        Lux.apply(layer, (1, 2, 3), Lux.setup(rng, layer)...)
        @test CNT[] == 2
        layer = Parallel(f_cnt, WrappedFunction(sin))
        Lux.apply(layer, 1, Lux.setup(rng, layer)...)
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
        ps, st = Lux.setup(rng, par)

        ip = Input(rand(Float32, 3, 3))
        ip2 = Input(rand(Float32, 3, 3))

        @test par(ip, ps, st)[1] ≈
            par.layers[1](ip.x, ps.layer_1, st.layer_1)[1] + par.layers[2](ip.x, ps.layer_2, st.layer_2)[1]
        @test par((ip, ip2), ps, st)[1] ≈
            par.layers[1](ip.x, ps.layer_1, st.layer_1)[1] + par.layers[2](ip2.x, ps.layer_2, st.layer_2)[1]
        gs = gradient((p, x...) -> sum(par(x, p, st)[1]), ps, ip, ip2)
        gs_reg = gradient(ps, ip, ip2) do p, x, y
            sum(par.layers[1](x.x, p.layer_1, st.layer_1)[1] + par.layers[2](y.x, p.layer_2, st.layer_2)[1])
        end

        @test gs[1] ≈ gs_reg[1]
        @test gs[2].x ≈ gs_reg[2].x
        @test gs[3].x ≈ gs_reg[3].x
    end
end

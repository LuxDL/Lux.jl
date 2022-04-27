@testset "Dense" begin
    @testset "constructors" begin
        layer = Dense(10, 100)
        ps, st = setup(MersenneTwister(0), layer)

        @test size(ps.weight) == (100, 10)
        @test size(ps.bias) == (100, 1)
        @test layer.λ == identity

        layer = Dense(10, 100, relu; bias=false)
        ps, st = setup(MersenneTwister(0), layer)

        @test !haskey(ps, :bias)
        @test layer.λ == relu
    end

    @testset "dimensions" begin
        layer = Dense(10, 5)
        ps, st = setup(MersenneTwister(0), layer)

        # For now only support 2D/1D input
        # @test length(first(apply(layer, randn(10), ps, st))) == 5
        # @test_throws DimensionMismatch first(apply(layer, randn(1), ps, st))
        # @test_throws MethodError first(apply(layer, 1, ps, st))
        @test size(first(apply(layer, randn(10), ps, st))) == (5,)
        @test size(first(apply(layer, randn(10, 2), ps, st))) == (5, 2)
        # @test size(first(apply(layer, randn(10, 2, 3), ps, st))) == (5, 2, 3)
        # @test size(first(apply(layer, randn(10, 2, 3, 4), ps, st))) == (5, 2, 3, 4)
        # @test_throws DimensionMismatch first(apply(layer, randn(11, 2, 3), ps, st))
    end

    @testset "zeros" begin
        @test begin
            layer = Dense(10, 1, identity; initW=ones)
            first(apply(layer, ones(10, 1), setup(MersenneTwister(0), layer)...))
        end == 10 * ones(1, 1)

        @test begin
            layer = Dense(10, 1, identity; initW=ones)
            first(apply(layer, ones(10, 2), setup(MersenneTwister(0), layer)...))
        end == 10 * ones(1, 2)

        @test begin
            layer = Dense(10, 2, identity; initW=ones)
            first(apply(layer, ones(10, 1), setup(MersenneTwister(0), layer)...))
        end == 10 * ones(2, 1)

        @test begin
            layer = Dense(10, 2, identity; initW=ones)
            first(apply(layer, [ones(10, 1) 2 * ones(10, 1)], setup(MersenneTwister(0), layer)...))
        end == [10 20; 10 20]

        @test begin
            layer = Dense(10, 2, identity; initW=ones, bias=false)
            first(apply(layer, [ones(10, 1) 2 * ones(10, 1)], setup(MersenneTwister(0), layer)...))
        end == [10 20; 10 20]
    end
end

@testset "SkipConnection" begin
    @testset "zero sum" begin
        input = randn(10, 10, 10, 10)
        layer = SkipConnection(WrappedFunction(zero), (a, b) -> a .+ b)
        @test apply(layer, input, setup(MersenneTwister(0), layer)...)[1] == input
    end

    @testset "concat size" begin
        input = randn(10, 2)
        layer = SkipConnection(Dense(10, 10), (a, b) -> hcat(a, b))
        @test size(apply(layer, input, setup(MersenneTwister(0), layer)...)[1]) == (10, 4)
    end
end

@testset "Parallel" begin
    @testset "zero sum" begin
        input = randn(10, 10, 10, 10)
        layer = Parallel(+, WrappedFunction(zero), NoOpLayer())
        @test layer(input, setup(MersenneTwister(0), layer)...)[1] == input
    end

    @testset "concat size" begin
        input = randn(10, 2)
        layer = Parallel((a, b) -> cat(a, b; dims=2), Dense(10, 10), NoOpLayer())
        @test size(apply(layer, input, setup(MersenneTwister(0), layer)...)[1]) == (10, 4)
        layer = Parallel(hcat, Dense(10, 10), NoOpLayer())
        @test size(apply(layer, input, setup(MersenneTwister(0), layer)...)[1]) == (10, 4)
    end

    @testset "vararg input" begin
        inputs = randn(10), randn(5), randn(4)
        layer = Parallel(+, Dense(10, 2), Dense(5, 2), Dense(4, 2))
        @test size(apply(layer, inputs, setup(MersenneTwister(0), layer)...)[1]) == (2,)
    end

    @testset "connection is called once" begin
        CNT = Ref(0)
        f_cnt = (x...) -> (CNT[] += 1; +(x...))
        layer = Parallel(f_cnt, WrappedFunction(sin), WrappedFunction(cos), WrappedFunction(tan))
        apply(layer, 1, setup(MersenneTwister(0), layer)...)
        @test CNT[] == 1
        apply(layer, (1, 2, 3), setup(MersenneTwister(0), layer)...)
        @test CNT[] == 2
        layer = Parallel(f_cnt, WrappedFunction(sin))
        apply(layer, 1, setup(MersenneTwister(0), layer)...)
        @test CNT[] == 3
    end

    # Ref https://github.com/FluxML/Flux.jl/issues/1673
    @testset "Input domain" begin
        struct Input
            x
        end

        struct L1 <: AbstractExplicitLayer end
        (::L1)(x, ps, st) = (ps.x * x, st)
        Lux.initialparameters(rng::AbstractRNG, ::L1) = (x=randn(rng, Float32, 3, 3),)
        Base.:*(a::AbstractArray, b::Input) = a * b.x

        par = Parallel(+, L1(), L1())
        ps, st = setup(MersenneTwister(0), par)

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

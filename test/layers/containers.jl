using Lux, NNlib, Random, Test, Zygote

include("../test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "SkipConnection" begin
    @testset "zero sum" begin
        layer = SkipConnection(WrappedFunction(zero), (a, b) -> a .+ b)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 10, 10, 10, 10)

        @test layer(x, ps, st)[1] == x
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3, reversediff_broken=true)
    end

    @testset "concat size" begin
        layer = SkipConnection(Dense(10, 10), (a, b) -> hcat(a, b))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 10, 2)

        @test size(layer(x, ps, st)[1]) == (10, 4)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end
end

@testset "Parallel" begin
    @testset "zero sum" begin
        layer = Parallel(+, WrappedFunction(zero), NoOpLayer())
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 10, 10, 10, 10)

        @test layer(x, ps, st)[1] == x
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3, reversediff_broken=true)
    end

    @testset "concat size" begin
        layer = Parallel((a, b) -> cat(a, b; dims=2), Dense(10, 10), NoOpLayer())
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = randn(rng, 10, 2)

        @test size(layer(x, ps, st)[1]) == (10, 4)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)

        layer = Parallel(hcat, Dense(10, 10), NoOpLayer())
        display(layer)
        ps, st = Lux.setup(rng, layer)

        @test size(layer(x, ps, st)[1]) == (10, 4)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)
    end

    @testset "vararg input" begin
        layer = Parallel(+, Dense(10, 2), Dense(5, 2), Dense(4, 2))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = (randn(rng, 10, 1), randn(rng, 5, 1), randn(rng, 4, 1))

        @test size(layer(x, ps, st)[1]) == (2, 1)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end

    @testset "named layers" begin
        layer = Parallel(+; d102=Dense(10, 2), d52=Dense(5, 2), d42=Dense(4, 2))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = (randn(rng, 10, 1), randn(rng, 5, 1), randn(rng, 4, 1))

        @test size(layer(x, ps, st)[1]) == (2, 1)
        run_JET_tests(layer, x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-3, rtol=1.0f-3)
    end

    @testset "connection is called once" begin
        CNT = Ref(0)
        f_cnt = (x...) -> (CNT[] += 1; +(x...))
        layer = Parallel(f_cnt, WrappedFunction(sin), WrappedFunction(cos),
                         WrappedFunction(tan))
        ps, st = Lux.setup(rng, layer)
        Lux.apply(layer, 1, ps, st)
        @test CNT[] == 1
        run_JET_tests(layer, 1, ps, st)
        Lux.apply(layer, (1, 2, 3), ps, st)
        @test CNT[] == 2
        layer = Parallel(f_cnt, WrappedFunction(sin))
        Lux.apply(layer, 1, ps, st)
        @test CNT[] == 3
    end

    # Ref https://github.com/FluxML/Flux.jl/issues/1673
    @testset "Input domain" begin
        struct Input
            x::Any
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
              par.layers[1](ip.x, ps.layer_1, st.layer_1)[1] +
              par.layers[2](ip.x, ps.layer_2, st.layer_2)[1]
        @test par((ip, ip2), ps, st)[1] ≈
              par.layers[1](ip.x, ps.layer_1, st.layer_1)[1] +
              par.layers[2](ip2.x, ps.layer_2, st.layer_2)[1]
        gs = Zygote.gradient((p, x...) -> sum(par(x, p, st)[1]), ps, ip, ip2)
        gs_reg = Zygote.gradient(ps, ip, ip2) do p, x, y
            return sum(par.layers[1](x.x, p.layer_1, st.layer_1)[1] +
                       par.layers[2](y.x, p.layer_2, st.layer_2)[1])
        end

        @test gs[1] ≈ gs_reg[1]
        @test gs[2].x ≈ gs_reg[2].x
        @test gs[3].x ≈ gs_reg[3].x
    end
end

@testset "PairwiseFusion" begin
    x = (rand(Float32, 1, 10), rand(Float32, 30, 10), rand(Float32, 10, 10))
    layer = PairwiseFusion(+, Dense(1, 30), Dense(30, 10))
    display(layer)
    ps, st = Lux.setup(rng, layer)
    y, _ = layer(x, ps, st)
    @test size(y) == (10, 10)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    layer = PairwiseFusion(+; d1=Dense(1, 30), d2=Dense(30, 10))
    display(layer)
    ps, st = Lux.setup(rng, layer)
    y, _ = layer(x, ps, st)
    @test size(y) == (10, 10)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    x = rand(1, 10)
    layer = PairwiseFusion(.+, Dense(1, 10), Dense(10, 1))
    display(layer)
    ps, st = Lux.setup(rng, layer)
    y, _ = layer(x, ps, st)
    @test size(y) == (1, 10)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    layer = PairwiseFusion(vcat, WrappedFunction(x -> x .+ 1), WrappedFunction(x -> x .+ 2),
                           WrappedFunction(x -> x .^ 3))
    display(layer)
    ps, st = Lux.setup(rng, layer)
    @test layer((2, 10, 20, 40), ps, st)[1] == [125, 1728, 8000, 40]

    layer = PairwiseFusion(vcat, WrappedFunction(x -> x .+ 1), WrappedFunction(x -> x .+ 2),
                           WrappedFunction(x -> x .^ 3))
    display(layer)
    ps, st = Lux.setup(rng, layer)
    @test layer(7, ps, st)[1] == [1000, 729, 343, 7]
end

@testset "BranchLayer" begin
    layer = BranchLayer(Dense(10, 10), Dense(10, 10))
    display(layer)
    ps, st = Lux.setup(rng, layer)
    x = rand(Float32, 10, 1)
    (y1, y2), _ = layer(x, ps, st)
    @test size(y1) == (10, 1)
    @test size(y2) == (10, 1)
    @test y1 == layer.layers.layer_1(x, ps.layer_1, st.layer_1)[1]
    @test y2 == layer.layers.layer_2(x, ps.layer_2, st.layer_2)[1]
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(sum, layer(x, ps, st)[1]), x, ps;
                                  atol=1.0f-3, rtol=1.0f-3)

    layer = BranchLayer(; d1=Dense(10, 10), d2=Dense(10, 10))
    display(layer)
    ps, st = Lux.setup(rng, layer)
    x = rand(Float32, 10, 1)
    (y1, y2), _ = layer(x, ps, st)
    @test size(y1) == (10, 1)
    @test size(y2) == (10, 1)
    @test y1 == layer.layers.d1(x, ps.d1, st.d1)[1]
    @test y2 == layer.layers.d2(x, ps.d2, st.d2)[1]
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(sum, layer(x, ps, st)[1]), x, ps;
                                  atol=1.0f-3, rtol=1.0f-3)
end

@testset "Chain" begin
    layer = Chain(Dense(10 => 5, sigmoid), Dense(5 => 2, tanh), Dense(2 => 1))
    display(layer)
    ps, st = Lux.setup(rng, layer)
    x = rand(Float32, 10, 1)
    y, _ = layer(x, ps, st)
    @test size(y) == (1, 1)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    layer = Chain(; l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1))
    display(layer)
    ps, st = Lux.setup(rng, layer)
    x = rand(Float32, 10, 1)
    y, _ = layer(x, ps, st)
    @test size(y) == (1, 1)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    layer = Chain(; l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1))
    display(layer)
    layer = layer[1:2]
    ps, st = Lux.setup(rng, layer)
    x = rand(Float32, 10, 1)
    y, _ = layer(x, ps, st)
    @test size(y) == (2, 1)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    layer = Chain(; l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1))
    display(layer)
    layer = layer[begin:(end - 1)]
    ps, st = Lux.setup(rng, layer)
    x = rand(Float32, 10, 1)
    y, _ = layer(x, ps, st)
    @test size(y) == (2, 1)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    layer = Chain(; l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh), d21=Dense(2 => 1))
    display(layer)
    layer = layer[1]
    ps, st = Lux.setup(rng, layer)
    x = rand(Float32, 10, 1)
    y, _ = layer(x, ps, st)
    @test size(y) == (5, 1)
    run_JET_tests(layer, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)

    @test_throws ArgumentError Chain(; l1=Dense(10 => 5, sigmoid), d52=Dense(5 => 2, tanh),
                                     d21=Dense(2 => 1), d2=Dense(2 => 1),
                                     disable_optimizations=false)
end

@testset "Maxout" begin
    @testset "constructor" begin
        layer = Maxout(() -> NoOpLayer(), 4)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = rand(rng, Float32, 10, 1)

        @test layer(x, ps, st)[1] == x
        run_JET_tests(layer, x, ps, st)
        @inferred layer(x, ps, st)
    end

    @testset "simple alternatives" begin
        layer = Maxout(NoOpLayer(), WrappedFunction(x -> 2x), WrappedFunction(x -> 0.5x))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = Float32.(collect(1:40))

        @test layer(x, ps, st)[1] == 2 .* x
        run_JET_tests(layer, x, ps, st)
        @inferred layer(x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)
    end

    @testset "complex alternatives" begin
        layer = Maxout(WrappedFunction(x -> [0.5; 0.1] * x),
                       WrappedFunction(x -> [0.2; 0.7] * x))
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = [3.0 2.0]
        y = [0.5, 0.7] .* x

        @test layer(x, ps, st)[1] == y
        run_JET_tests(layer, x, ps, st)
        @inferred layer(x, ps, st)
        test_gradient_correctness_fdm(x -> sum(layer(x, ps, st)[1]), x; atol=1.0f-3,
                                      rtol=1.0f-3)
    end

    @testset "params" begin
        layer = Maxout(() -> Dense(2, 4), 4)
        display(layer)
        ps, st = Lux.setup(rng, layer)
        x = [10.0f0 3.0f0]'

        @test Lux.parameterlength(layer) == sum(Lux.parameterlength.(values(layer.layers)))
        @test size(layer(x, ps, st)[1]) == (4, 1)
        run_JET_tests(layer, x, ps, st)
        @inferred layer(x, ps, st)
        test_gradient_correctness_fdm((x, ps) -> sum(layer(x, ps, st)[1]), x, ps;
                                      atol=1.0f-1, rtol=1.0f-1)
    end
end

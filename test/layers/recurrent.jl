using JET, Lux, NNlib, Random, Test

include("../test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "RNNCell" begin
    for rnncell in (RNNCell(3 => 5, identity), RNNCell(3 => 5, tanh),
                    RNNCell(3 => 5, tanh; use_bias=false),
                    RNNCell(3 => 5, identity; use_bias=false))
        println(rnncell)
        ps, st = Lux.setup(rng, rnncell)
        x = randn(rng, Float32, 3, 2)
        h, st_ = Lux.apply(rnncell, x, ps, st)

        run_JET_tests(rnncell, x, ps, st)
        run_JET_tests(rnncell, (x, h), ps, st_)

        function loss_loop_rnncell(p)
            h, st_ = rnncell(x, p, st)
            for i in 1:10
                h, st_ = rnncell((x, h), p, st_)
            end
            return sum(abs2, h)
        end

        test_gradient_correctness_fdm(loss_loop_rnncell, ps; atol=1e-3, rtol=1e-3)
    end
    # Deprecated Functionality (Remove in v0.5)
    @testset "Deprecations" begin
        @test_deprecated RNNCell(3 => 5, relu; bias=false)
        @test_deprecated RNNCell(3 => 5, relu; bias=true)
        @test_throws ArgumentError RNNCell(3 => 5, relu; bias=false, use_bias=false)
    end
end

@testset "LSTMCell" begin for lstmcell in (LSTMCell(3 => 5),
                                           LSTMCell(3 => 5; use_bias=true),
                                           LSTMCell(3 => 5; use_bias=false))
    println(lstmcell)
    ps, st = Lux.setup(rng, lstmcell)
    x = randn(rng, Float32, 3, 2)
    (h, c), st_ = Lux.apply(lstmcell, x, ps, st)

    run_JET_tests(lstmcell, x, ps, st)
    run_JET_tests(lstmcell, (x, h, c), ps, st_)

    function loss_loop_lstmcell(p)
        (h, c), st_ = lstmcell(x, p, st)
        for i in 1:10
            (h, c), st_ = lstmcell((x, h, c), p, st_)
        end
        return sum(abs2, h)
    end

    test_gradient_correctness_fdm(loss_loop_lstmcell, ps; atol=1e-3, rtol=1e-3)
end end

@testset "GRUCell" begin for grucell in (GRUCell(3 => 5), GRUCell(3 => 5; use_bias=true),
                                         GRUCell(3 => 5; use_bias=false))
    println(grucell)
    ps, st = Lux.setup(rng, grucell)
    x = randn(rng, Float32, 3, 2)
    h, st_ = Lux.apply(grucell, x, ps, st)

    run_JET_tests(grucell, x, ps, st)
    run_JET_tests(grucell, (x, h), ps, st_)

    function loss_loop_grucell(p)
        h, st_ = grucell(x, p, st)
        for i in 1:10
            h, st_ = grucell((x, h), p, st_)
        end
        return sum(abs2, h)
    end

    test_gradient_correctness_fdm(loss_loop_grucell, ps; atol=1e-3, rtol=1e-3)
end end

@testset "multigate" begin
    x = rand(6, 5)
    res, (dx,) = Zygote.withgradient(x) do x
        x1, _, x3 = Lux.multigate(x, Val(3))
        return sum(x1) + sum(x3 .* 2)
    end
    @test res == sum(x[1:2, :]) + 2sum(x[5:6, :])
    @test dx == [ones(2, 5); zeros(2, 5); fill(2, 2, 5)]
end

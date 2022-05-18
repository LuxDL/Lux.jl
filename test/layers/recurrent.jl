@testset "Recurrent Cell Implementations" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    @testset "RNNCell" begin
        for rnncell in (
            RNNCell(3 => 5, identity),
            RNNCell(3 => 5, tanh),
            RNNCell(3 => 5, tanh; bias=false),
            RNNCell(3 => 5, identity; bias=false),
        )
            ps, st = Lux.setup(rng, rnncell)
            x = randn(rng, Float32, 3, 2)
            h, st_ = Lux.apply(rnncell, x, ps, st)

            @test_call rnncell(x, ps, st)
            @test_opt target_modules = (Lux,) rnncell(x, ps, st)
            @test_call rnncell((x, h), ps, st_)
            @test_opt target_modules = (Lux,) rnncell((x, h), ps, st_)
        end
    end

    @testset "LSTMCell" begin
        lstmcell = LSTMCell(3 => 5)
        ps, st = Lux.setup(rng, lstmcell)
        x = randn(rng, Float32, 3, 2)
        (h, c), st_ = Lux.apply(lstmcell, x, ps, st)

        @test_call lstmcell(x, ps, st)
        @test_opt target_modules = (Lux,) lstmcell(x, ps, st)
        @test_call lstmcell((x, h, c), ps, st_)
        @test_opt target_modules = (Lux,) lstmcell((x, h, c), ps, st_)
    end

    @testset "GRUCell" begin
        grucell = GRUCell(3 => 5)
        ps, st = Lux.setup(rng, grucell)
        x = randn(rng, Float32, 3, 2)
        h, st_ = Lux.apply(grucell, x, ps, st)

        @test_call grucell(x, ps, st)
        @test_opt target_modules = (Lux,) grucell(x, ps, st)
        @test_call grucell((x, h), ps, st_)
        @test_opt target_modules = (Lux,) grucell((x, h), ps, st_)
    end
end

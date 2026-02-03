using LuxTestUtils: check_approx

include("../shared_testsetup.jl")

# Helper functions
function loss_loop(cell, x, p, st)
    (y, carry), st_ = cell(x, p, st)
    for _ in 1:3
        (y, carry), st_ = cell((x, carry), p, st_)
    end
    return sum(abs2, y)
end

@testset "LSTMCell" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset for use_bias in (true, false)
            lstmcell = LSTMCell(3 => 5; use_bias)
            display(lstmcell)
            ps, st = dev(Lux.setup(rng, lstmcell))

            @testset for x_size in ((3, 2), (3,))
                x = aType(randn(rng, Float32, x_size...))
                (y, carry), st_ = Lux.apply(lstmcell, x, ps, st)

                @jet lstmcell(x, ps, st)
                @jet lstmcell((x, carry), ps, st)

                @test !hasproperty(ps, :hidden_state)
                @test !hasproperty(ps, :memory)

                @test_gradients(loss_loop, lstmcell, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
            end
        end

        @testset "Trainable hidden states" begin
            @testset for x_size in ((3, 2), (3,))
                x = aType(randn(rng, Float32, x_size...))
                _lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=false, train_memory=false
                )
                _ps, _st = dev(Lux.setup(rng, _lstm))
                (_y, _carry), _ = Lux.apply(_lstm, x, _ps, _st)

                lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=false, train_memory=false
                )
                ps, st = dev(Lux.setup(rng, lstm))
                ps = _ps
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry

                if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
                    l, back = Zygote.pullback(
                        p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps
                    )
                    gs = back(one(l))[1]

                    @test !hasproperty(gs, :bias_ih)
                    @test !hasproperty(gs, :bias_hh)
                    @test !hasproperty(gs, :hidden_state)
                    @test !hasproperty(gs, :memory)
                end

                lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=true, train_memory=false
                )
                ps, st = dev(Lux.setup(rng, lstm))
                ps = merge(_ps, (hidden_state=ps.hidden_state,))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry

                if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
                    l, back = Zygote.pullback(
                        p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps
                    )
                    gs = back(one(l))[1]
                    @test !hasproperty(gs, :bias_ih)
                    @test !hasproperty(gs, :bias_hh)
                    @test !isnothing(gs.hidden_state)
                    @test !hasproperty(gs, :memory)
                end

                lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=false, train_memory=true
                )
                ps, st = dev(Lux.setup(rng, lstm))
                ps = merge(_ps, (memory=ps.memory,))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry

                if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
                    l, back = Zygote.pullback(
                        p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps
                    )
                    gs = back(one(l))[1]
                    @test !hasproperty(gs, :bias_ih)
                    @test !hasproperty(gs, :bias_hh)
                    @test !hasproperty(gs, :hidden_state)
                    @test !isnothing(gs.memory)
                end

                lstm = LSTMCell(3 => 5; use_bias=false, train_state=true, train_memory=true)
                ps, st = dev(Lux.setup(rng, lstm))
                ps = merge(_ps, (hidden_state=ps.hidden_state, memory=ps.memory))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry

                if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
                    l, back = Zygote.pullback(
                        p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps
                    )
                    gs = back(one(l))[1]
                    @test !hasproperty(gs, :bias_ih)
                    @test !hasproperty(gs, :bias_hh)
                    @test !isnothing(gs.hidden_state)
                    @test !isnothing(gs.memory)
                end

                lstm = LSTMCell(3 => 5; use_bias=true, train_state=true, train_memory=true)
                ps, st = dev(Lux.setup(rng, lstm))
                ps = merge(_ps, (; ps.bias_ih, ps.bias_hh, ps.hidden_state, ps.memory))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)

                if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
                    l, back = Zygote.pullback(
                        p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps
                    )
                    gs = back(one(l))[1]
                    @test !isnothing(gs.bias_ih)
                    @test !isnothing(gs.bias_hh)
                    @test !isnothing(gs.hidden_state)
                    @test !isnothing(gs.memory)
                end
            end
        end
    end
end

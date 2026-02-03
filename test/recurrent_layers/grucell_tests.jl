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

@testset "GRUCell" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset for use_bias in (true, false)
            grucell = GRUCell(3 => 5; use_bias)
            display(grucell)
            ps, st = dev(Lux.setup(rng, grucell))

            @testset for x_size in ((3, 2), (3,))
                x = aType(randn(rng, Float32, x_size...))
                (y, carry), st_ = Lux.apply(grucell, x, ps, st)

                @jet grucell(x, ps, st)
                @jet grucell((x, carry), ps, st)

                @test !hasproperty(ps, :hidden_state)

                @test_gradients(loss_loop, grucell, x, ps, st; atol=1.0e-3, rtol=1.0e-3)
            end
        end

        @testset "Trainable hidden states" begin
            @testset for x_size in ((3, 2), (3,))
                x = aType(randn(rng, Float32, x_size...))
                _gru = GRUCell(3 => 5; use_bias=false, train_state=false)
                _ps, _st = dev(Lux.setup(rng, _gru))
                (_y, _carry), _ = Lux.apply(_gru, x, _ps, _st)

                gru = GRUCell(3 => 5; use_bias=false, train_state=false)
                ps, st = dev(Lux.setup(rng, gru))
                ps = _ps
                (y, carry), _ = Lux.apply(gru, x, ps, st)
                @test check_approx(carry, _carry)

                if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
                    l, back = Zygote.pullback(
                        p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps
                    )
                    gs = back(one(l))[1]

                    @test !hasproperty(gs, :bias_ih)
                    @test !hasproperty(gs, :bias_hh)
                    @test !hasproperty(gs, :hidden_state)
                end

                gru = GRUCell(3 => 5; use_bias=false, train_state=true)
                ps, st = dev(Lux.setup(rng, gru))
                ps = merge(_ps, (hidden_state=ps.hidden_state,))
                (y, carry), _ = Lux.apply(gru, x, ps, st)
                @test check_approx(carry, _carry)

                if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
                    l, back = Zygote.pullback(
                        p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps
                    )
                    gs = back(one(l))[1]
                    @test !isnothing(gs.hidden_state)
                end

                gru = GRUCell(3 => 5; use_bias=true, train_state=true)
                ps, st = dev(Lux.setup(rng, gru))
                ps = merge(_ps, (; ps.bias_ih, ps.bias_hh, ps.hidden_state))
                (y, carry), _ = Lux.apply(gru, x, ps, st)
                @test check_approx(carry, _carry)

                if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
                    l, back = Zygote.pullback(
                        p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps
                    )
                    gs = back(one(l))[1]
                    @test !isnothing(gs.hidden_state)
                end
            end
        end
    end
end

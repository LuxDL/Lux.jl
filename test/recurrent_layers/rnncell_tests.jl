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

@testset "RNNCell" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset for act in (identity, tanh),
            use_bias in (true, false),
            train_state in (true, false)

            rnncell = RNNCell(3 => 5, act; use_bias, train_state)
            display(rnncell)
            ps, st = dev(Lux.setup(rng, rnncell))
            @testset for x_size in ((3, 2), (3,))
                x = aType(randn(rng, Float32, x_size...))
                (y, carry), st_ = Lux.apply(rnncell, x, ps, st)

                @jet rnncell(x, ps, st)
                @jet rnncell((x, carry), ps, st)

                if train_state
                    @test hasproperty(ps, :hidden_state)
                else
                    @test !hasproperty(ps, :hidden_state)
                end

                @test_gradients(loss_loop, rnncell, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
            end
        end

        @testset "Trainable hidden states" begin
            @testset for use_bias in (true, false)
                rnncell = RNNCell(3 => 5, identity; use_bias, train_state=true)
                rnn_no_trainable_state = RNNCell(
                    3 => 5, identity; use_bias=false, train_state=false
                )
                _ps, _st = dev(Lux.setup(rng, rnn_no_trainable_state))

                ps, st = dev(Lux.setup(rng, rnncell))
                ps = merge(_ps, (hidden_state=ps.hidden_state,))

                @testset for x_size in ((3, 2), (3,))
                    x = aType(randn(rng, Float32, x_size...))
                    (_y, _carry), _ = Lux.apply(rnn_no_trainable_state, x, _ps, _st)

                    (y, carry), _ = Lux.apply(rnncell, x, ps, st)
                    @test carry == _carry

                    if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
                        l, back = Zygote.pullback(
                            p -> sum(abs2, 0 .- rnncell(x, p, st)[1][1]), ps
                        )
                        gs = back(one(l))[1]
                        @test !isnothing(gs.hidden_state)
                    end
                end
            end
        end
    end
end

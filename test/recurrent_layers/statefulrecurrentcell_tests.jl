using LuxTestUtils: check_approx

include("../shared_testsetup.jl")

# Helper functions
function loss_loop_no_carry(cell, x, p, st)
    y, st_ = cell(x, p, st)
    for i in 1:3
        y, st_ = cell(x, p, st_)
    end
    return sum(abs2, y)
end

@testset "StatefulRecurrentCell" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset for _cell in (RNNCell, LSTMCell, GRUCell),
            use_bias in (true, false),
            train_state in (true, false)

            cell = _cell(3 => 5; use_bias, train_state)
            rnn = StatefulRecurrentCell(cell)
            display(rnn)

            @testset for x_size in ((3, 2), (3,))
                x = aType(randn(rng, Float32, x_size...))
                ps, st = dev(Lux.setup(rng, rnn))

                y, st_ = rnn(x, ps, st)

                @jet rnn(x, ps, st)
                @jet rnn(x, ps, st_)

                if length(x_size) == 2
                    @test size(y) == (5, 2)
                else
                    @test size(y) == (5,)
                end
                @test st.carry === nothing
                @test st_.carry !== nothing

                st__ = Lux.update_state(st, :carry, nothing)
                @test st__.carry === nothing

                @test_gradients(
                    loss_loop_no_carry,
                    rnn,
                    x,
                    ps,
                    st;
                    atol=1.0e-3,
                    rtol=1.0e-3,
                    soft_fail=[AutoFiniteDiff()]
                )
            end
        end
    end
end

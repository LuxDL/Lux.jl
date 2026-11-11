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

    @testset "StatefulLuxLayer with StatefulRecurrentCell" begin
        @test !LuxCore.preserves_state_type(StatefulRecurrentCell(RNNCell(2 => 3)))
        @test !LuxCore.preserves_state_type(
            Chain(StatefulRecurrentCell(RNNCell(2 => 3)), Dense(3 => 2))
        )

        @testset for _cell in (RNNCell, LSTMCell, GRUCell)
            cell = _cell(2 => 3)
            rcell = StatefulRecurrentCell(cell)
            @test !LuxCore.preserves_state_type(rcell)
            @test !LuxCore.preserves_state_type(Chain(rcell, Dense(3 => 2)))

            ps, st = Lux.setup(rng, rcell)
            smodel = StatefulLuxLayer(rcell, ps, st)
            @test smodel.fixed_state_type === Val(false)

            x1 = randn(rng, Float32, 2, 4)
            x2 = randn(rng, Float32, 2, 4)
            st_init = smodel.st === nothing ? smodel.st_any : smodel.st
            @test st_init.carry === nothing

            y1 = smodel(x1)
            st_step1 = smodel.st === nothing ? smodel.st_any : smodel.st
            @test st_step1.carry !== nothing
            c1 = deepcopy(st_step1.carry)

            y2 = smodel(x2)
            st_step2 = smodel.st === nothing ? smodel.st_any : smodel.st
            @test st_step2.carry !== nothing
            @test st_step2.carry != c1

            # Test within a Chain
            chain = Chain(rcell, Dense(3 => 2))
            ps_c, st_c = Lux.setup(rng, chain)
            schain = StatefulLuxLayer(chain, ps_c, st_c)
            @test schain.fixed_state_type === Val(false)

            y_c1 = schain(x1)
            st_c1 = schain.st === nothing ? schain.st_any : schain.st
            @test st_c1.layer_1.carry !== nothing
            c_c1 = deepcopy(st_c1.layer_1.carry)

            y_c2 = schain(x2)
            st_c2 = schain.st === nothing ? schain.st_any : schain.st
            @test st_c2.layer_1.carry !== nothing
            @test st_c2.layer_1.carry != c_c1
        end
    end
end

using LuxTestUtils: check_approx

include("../shared_testsetup.jl")

@testset "Bidirectional" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        cell_configs = (
            (RNNCell, nothing),
            (LSTMCell, nothing),
            (GRUCell, nothing),
            (RNNCell, LSTMCell),
            (LSTMCell, GRUCell),
            (GRUCell, RNNCell),
        )
        @testset "cells: ($_fcell, $_bcell)" for (_fcell, _bcell) in cell_configs
            cell = _fcell(3 => 5)
            backward_cell = _bcell === nothing ? nothing : _bcell(3 => 5)
            bi_rnn = BidirectionalRNN(cell, backward_cell)
            bi_rnn_no_merge = BidirectionalRNN(cell, backward_cell; merge_mode=nothing)
            display(bi_rnn)

            # Batched Time Series
            x = aType(randn(rng, Float32, 3, 4, 2))
            ps, st = dev(Lux.setup(rng, bi_rnn))
            y, st_ = bi_rnn(x, ps, st)
            y_, st__ = bi_rnn_no_merge(x, ps, st)

            @jet bi_rnn(x, ps, st)

            @test size(y) == (4,)
            @test all(x -> size(x) == (10, 2), y)

            @test length(y_) == 2
            @test size(y_[1]) == size(y_[2])
            @test size(y_[1]) == (4,)
            @test all(x -> size(x) == (5, 2), y_[1])

            @test_gradients(
                sumsumfirst,
                bi_rnn,
                x,
                ps,
                st;
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )

            __f =
                (bi_rnn_no_merge, x, ps, st) -> begin
                    (y1, y2), st_ = bi_rnn_no_merge(x, ps, st)
                    return sum(Base.Fix1(sum, abs2), y1) + sum(Base.Fix1(sum, abs2), y2)
                end
            @test_gradients(
                __f,
                bi_rnn_no_merge,
                x,
                ps,
                st;
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
        end
    end
end

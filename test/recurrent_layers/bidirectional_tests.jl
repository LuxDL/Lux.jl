using LuxTestUtils: check_approx

include("../shared_testsetup.jl")

@testset "Bidirectional" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "cell: $_cell" for _cell in (RNNCell, LSTMCell, GRUCell)
            cell = _cell(3 => 5)
            bi_rnn = BidirectionalRNN(cell)
            bi_rnn_no_merge = BidirectionalRNN(cell; merge_mode=nothing)
            display(bi_rnn)

            # Batched Time Series
            x = aType(randn(rng, Float32, 3, 4, 2))
            ps, st = dev(Lux.setup(rng, bi_rnn))
            y, st_ = bi_rnn(x, ps, st)
            y_, st__ = bi_rnn_no_merge(x, ps, st)

            @jet bi_rnn(x, ps, st)
            @jet bi_rnn_no_merge(x, ps, st)

            @test size(y) == (4,)
            @test all(x -> size(x) == (10, 2), y)

            @test length(y_) == 2
            @test size(y_[1]) == size(y_[2])
            @test size(y_[1]) == (4,)
            @test all(x -> size(x) == (5, 2), y_[1])

            # Check batch consistency: individual batch slices match batched output
            for b in 1:2
                x_b = x[:, :, b:b]
                y_b, _ = bi_rnn(x_b, ps, st)
                for t in 1:4
                    @test check_approx(y_b[t][:, 1], y[t][:, b])
                end

                x_mat = x[:, :, b]
                y_mat, _ = bi_rnn(x_mat, ps, st)
                for t in 1:4
                    @test check_approx(y_mat[t], y[t][:, b])
                end
            end

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

            @testset "ordering: TimeLastIndex()" begin
                bi_rnn_tli = BidirectionalRNN(cell; ordering=TimeLastIndex())
                bi_rnn_no_merge_tli = BidirectionalRNN(
                    cell; merge_mode=nothing, ordering=TimeLastIndex()
                )
                display(bi_rnn_tli)

                # Batched Time Series (features, batch, time)
                x_tli = aType(randn(rng, Float32, 3, 2, 4))
                ps_tli, st_tli = dev(Lux.setup(rng, bi_rnn_tli))
                y_tli, st_tli_ = bi_rnn_tli(x_tli, ps_tli, st_tli)
                y_tli_, st_tli__ = bi_rnn_no_merge_tli(x_tli, ps_tli, st_tli)

                @jet bi_rnn_tli(x_tli, ps_tli, st_tli)
                @jet bi_rnn_no_merge_tli(x_tli, ps_tli, st_tli)

                @test size(y_tli) == (4,)
                @test all(x -> size(x) == (10, 2), y_tli)

                @test length(y_tli_) == 2
                @test size(y_tli_[1]) == size(y_tli_[2])
                @test size(y_tli_[1]) == (4,)
                @test all(x -> size(x) == (5, 2), y_tli_[1])

                for b in 1:2
                    x_b = x_tli[:, b:b, :]
                    y_b, _ = bi_rnn_tli(x_b, ps_tli, st_tli)
                    for t in 1:4
                        @test check_approx(y_b[t][:, 1], y_tli[t][:, b])
                    end

                    x_mat = x_tli[:, b, :]
                    y_mat, _ = bi_rnn_tli(x_mat, ps_tli, st_tli)
                    for t in 1:4
                        @test check_approx(y_mat[t], y_tli[t][:, b])
                    end
                end

                @test_gradients(
                    sumsumfirst,
                    bi_rnn_tli,
                    x_tli,
                    ps_tli,
                    st_tli;
                    atol=1.0e-3,
                    rtol=1.0e-3,
                    skip_backends=[AutoEnzyme()]
                )

                __f_tli =
                    (bi_rnn_no_merge_tli, x, ps, st) -> begin
                        (y1, y2), st_ = bi_rnn_no_merge_tli(x, ps, st)
                        return sum(Base.Fix1(sum, abs2), y1) + sum(Base.Fix1(sum, abs2), y2)
                    end
                @test_gradients(
                    __f_tli,
                    bi_rnn_no_merge_tli,
                    x_tli,
                    ps_tli,
                    st_tli;
                    atol=1.0e-3,
                    rtol=1.0e-3,
                    skip_backends=[AutoEnzyme()]
                )
            end

            @testset "Vector and Tuple inputs" begin
                x_vec = aType.([randn(rng, Float32, 3, 2) for _ in 1:4])
                y_vec, _ = bi_rnn(x_vec, ps, st)
                @test length(y_vec) == 4
                @test all(x -> size(x) == (10, 2), y_vec)

                x_tup = Tuple(x_vec)
                y_tup, _ = bi_rnn(x_tup, ps, st)
                @test length(y_tup) == 4
                for t in 1:4
                    @test check_approx(y_tup[t], y_vec[t])
                end
            end

            @testset for _backward_cell in (RNNCell, LSTMCell, GRUCell)
                cell = _cell(3 => 5)
                backward_cell = _backward_cell(3 => 5)
                bi_rnn = BidirectionalRNN(cell, backward_cell)
                bi_rnn_no_merge = BidirectionalRNN(cell, backward_cell; merge_mode=nothing)
                display(bi_rnn)

                # Batched Time Series
                x = aType(randn(rng, Float32, 3, 4, 2))
                ps, st = dev(Lux.setup(rng, bi_rnn))
                y, st_ = bi_rnn(x, ps, st)
                y_, st__ = bi_rnn_no_merge(x, ps, st)

                @jet bi_rnn(x, ps, st)
                @jet bi_rnn_no_merge(x, ps, st)

                @test size(y) == (4,)
                @test all(x -> size(x) == (10, 2), y)

                @test length(y_) == 2
                @test size(y_[1]) == size(y_[2])
                @test size(y_[1]) == (4,)
                @test all(x -> size(x) == (5, 2), y_[1])

                for b in 1:2
                    x_b = x[:, :, b:b]
                    y_b, _ = bi_rnn(x_b, ps, st)
                    for t in 1:4
                        @test check_approx(y_b[t][:, 1], y[t][:, b])
                    end
                end

                # Verify TimeLastIndex() with distinct backward_cell
                bi_rnn_tli = BidirectionalRNN(
                    cell, backward_cell; ordering=TimeLastIndex()
                )
                ps_tli, st_tli = dev(Lux.setup(rng, bi_rnn_tli))
                x_tli = aType(randn(rng, Float32, 3, 2, 4))
                y_tli, _ = bi_rnn_tli(x_tli, ps_tli, st_tli)
                for b in 1:2
                    x_b = x_tli[:, b:b, :]
                    y_b, _ = bi_rnn_tli(x_b, ps_tli, st_tli)
                    for t in 1:4
                        @test check_approx(y_b[t][:, 1], y_tli[t][:, b])
                    end
                end

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
end

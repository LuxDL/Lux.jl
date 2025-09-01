@testsetup module RecurrentLayersSetup

function loss_loop(cell, x, p, st)
    (y, carry), st_ = cell(x, p, st)
    for _ in 1:3
        (y, carry), st_ = cell((x, carry), p, st_)
    end
    return sum(abs2, y)
end

function loss_loop_no_carry(cell, x, p, st)
    y, st_ = cell(x, p, st)
    for i in 1:3
        y, st_ = cell(x, p, st_)
    end
    return sum(abs2, y)
end

export loss_loop, loss_loop_no_carry

end

@testitem "RNNCell" setup = [SharedTestSetup, RecurrentLayersSetup] tags = [
    :recurrent_layers
] begin
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

@testitem "LSTMCell" setup = [SharedTestSetup, RecurrentLayersSetup] tags = [
    :recurrent_layers
] begin
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
                l, back = Zygote.pullback(
                    p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps
                )
                gs = back(one(l))[1]

                @test !hasproperty(gs, :bias_ih)
                @test !hasproperty(gs, :bias_hh)
                @test !hasproperty(gs, :hidden_state)
                @test !hasproperty(gs, :memory)

                lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=true, train_memory=false
                )
                ps, st = dev(Lux.setup(rng, lstm))
                ps = merge(_ps, (hidden_state=ps.hidden_state,))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(
                    p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps
                )
                gs = back(one(l))[1]
                @test !hasproperty(gs, :bias_ih)
                @test !hasproperty(gs, :bias_hh)
                @test !isnothing(gs.hidden_state)
                @test !hasproperty(gs, :memory)

                lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=false, train_memory=true
                )
                ps, st = dev(Lux.setup(rng, lstm))
                ps = merge(_ps, (memory=ps.memory,))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(
                    p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps
                )
                gs = back(one(l))[1]
                @test !hasproperty(gs, :bias_ih)
                @test !hasproperty(gs, :bias_hh)
                @test !hasproperty(gs, :hidden_state)
                @test !isnothing(gs.memory)

                lstm = LSTMCell(3 => 5; use_bias=false, train_state=true, train_memory=true)
                ps, st = dev(Lux.setup(rng, lstm))
                ps = merge(_ps, (hidden_state=ps.hidden_state, memory=ps.memory))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(
                    p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps
                )
                gs = back(one(l))[1]
                @test !hasproperty(gs, :bias_ih)
                @test !hasproperty(gs, :bias_hh)
                @test !isnothing(gs.hidden_state)
                @test !isnothing(gs.memory)

                lstm = LSTMCell(3 => 5; use_bias=true, train_state=true, train_memory=true)
                ps, st = dev(Lux.setup(rng, lstm))
                ps = merge(_ps, (; ps.bias_ih, ps.bias_hh, ps.hidden_state, ps.memory))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
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

@testitem "GRUCell" setup = [SharedTestSetup, RecurrentLayersSetup] tags = [
    :recurrent_layers
] begin
    using LuxTestUtils: check_approx

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
                l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]

                @test !hasproperty(gs, :bias_ih)
                @test !hasproperty(gs, :bias_hh)
                @test !hasproperty(gs, :hidden_state)

                gru = GRUCell(3 => 5; use_bias=false, train_state=true)
                ps, st = dev(Lux.setup(rng, gru))
                ps = merge(_ps, (hidden_state=ps.hidden_state,))
                (y, carry), _ = Lux.apply(gru, x, ps, st)
                @test check_approx(carry, _carry)
                l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test !isnothing(gs.hidden_state)

                gru = GRUCell(3 => 5; use_bias=true, train_state=true)
                ps, st = dev(Lux.setup(rng, gru))
                ps = merge(_ps, (; ps.bias_ih, ps.bias_hh, ps.hidden_state))
                (y, carry), _ = Lux.apply(gru, x, ps, st)
                @test check_approx(carry, _carry)
                l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test !isnothing(gs.hidden_state)
            end
        end
    end
end

@testitem "StatefulRecurrentCell" setup = [SharedTestSetup, RecurrentLayersSetup] tags = [
    :recurrent_layers
] begin
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

@testsetup module RecurrenceTestSetup

using LuxTestUtils, StableRNGs, Test, Lux

sumabs2first(layer, x, ps, st) = sum(abs2, first(layer(x, ps, st)))
sumsumfirst(layer, x, ps, st) = sum(sum, first(layer(x, ps, st)))

function test_recurrence_layer(
    mode, aType, dev, ongpu, ordering, _cell, use_bias, train_state
)
    rng = StableRNG(12345)

    cell = _cell(3 => 5; use_bias, train_state)
    rnn = Recurrence(cell; ordering)
    display(rnn)
    rnn_seq = Recurrence(cell; ordering, return_sequence=true)
    display(rnn_seq)

    # Batched Time Series
    @testset "typeof(x): $(typeof(x))" for x in (
        aType(randn(rng, Float32, 3, 4, 2)),
        # aType.(Tuple(randn(rng, Float32, 3, 2) for _ in 1:4)),
        aType.([randn(rng, Float32, 3, 2) for _ in 1:4]),
    )
        # Fix data ordering for testing
        if ordering isa TimeLastIndex && x isa AbstractArray && ndims(x) ≥ 2
            x = permutedims(x, (ntuple(identity, ndims(x) - 2)..., ndims(x), ndims(x) - 1))
        end

        ps, st = dev(Lux.setup(rng, rnn))
        y, st_ = rnn(x, ps, st)
        y_, st__ = rnn_seq(x, ps, st)

        @test size(y) == (5, 2)
        @test length(y_) == 4
        @test all(x -> size(x) == (5, 2), y_)

        @test_gradients(
            sumabs2first,
            rnn,
            x,
            ps,
            st;
            atol=1.0f-3,
            rtol=1.0f-3,
            skip_backends=[AutoEnzyme()],
            soft_fail=[AutoFiniteDiff()]
        )

        @test_gradients(
            sumsumfirst,
            rnn_seq,
            x,
            ps,
            st;
            atol=1.0f-3,
            rtol=1.0f-3,
            skip_backends=[AutoEnzyme()],
            soft_fail=[AutoFiniteDiff()]
        )
    end

    # Batched Time Series without data batches
    @testset "typeof(x): $(typeof(x))" for x in (
        aType(randn(rng, Float32, 3, 4)),
        # aType.(Tuple(randn(rng, Float32, 3) for _ in 1:4)),
        aType.([randn(rng, Float32, 3) for _ in 1:4]),
    )
        ps, st = dev(Lux.setup(rng, rnn))
        y, st_ = rnn(x, ps, st)
        y_, st__ = rnn_seq(x, ps, st)

        @test size(y) == (5,)
        @test length(y_) == 4
        @test all(x -> size(x) == (5,), y_)

        if x isa AbstractMatrix && ordering isa BatchLastIndex
            x2 = reshape(x, Val(3))
            y2, _ = rnn(x2, ps, st)
            @test y == vec(y2)
            y2_, _ = rnn_seq(x2, ps, st)
            @test all(x -> x[1] == vec(x[2]), zip(y_, y2_))
        end

        @test_gradients(
            sumabs2first,
            rnn,
            x,
            ps,
            st;
            atol=1.0f-3,
            rtol=1.0f-3,
            skip_backends=[AutoEnzyme()],
            soft_fail=[AutoFiniteDiff()]
        )

        @test_gradients(
            sumsumfirst,
            rnn_seq,
            x,
            ps,
            st;
            atol=1.0f-3,
            rtol=1.0f-3,
            skip_backends=[AutoEnzyme()],
            soft_fail=[AutoFiniteDiff()]
        )
    end
end

const ALL_TEST_CONFIGS = Iterators.product(
    (BatchLastIndex(), TimeLastIndex()),
    (RNNCell, LSTMCell, GRUCell),
    (true, false),
    (true, false),
)

const TEST_BLOCKS = collect(
    Iterators.partition(ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 2))
)

export TEST_BLOCKS, test_recurrence_layer

end

@testitem "Recurrence: Group 1" setup = [
    RecurrenceTestSetup, SharedTestSetup, RecurrentLayersSetup
] tags = [:recurrent_layers] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset for (ordering, cell, use_bias, train_state) in TEST_BLOCKS[1]
            test_recurrence_layer(
                mode, aType, dev, ongpu, ordering, cell, use_bias, train_state
            )
        end
    end
end

@testitem "Recurrence: Group 2" setup = [
    RecurrenceTestSetup, SharedTestSetup, RecurrentLayersSetup
] tags = [:recurrent_layers] begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset for (ordering, cell, use_bias, train_state) in TEST_BLOCKS[2]
            test_recurrence_layer(
                mode, aType, dev, ongpu, ordering, cell, use_bias, train_state
            )
        end
    end
end

@testitem "Recurrence Ordering Check #302" setup = [SharedTestSetup] tags = [
    :recurrent_layers
] begin
    rng = StableRNG(12345)
    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        encoder = Recurrence(
            RNNCell(
                1 => 1,
                identity;
                init_weight=(rng, args...; kwargs...) -> ones(args...; kwargs...),
                init_state=(rng, args...; kwargs...) -> zeros(args...; kwargs...),
                init_bias=(rng, args...; kwargs...) -> zeros(args...; kwargs...),
            );
            return_sequence=true,
        )
        ps, st = dev(Lux.setup(rng, encoder))
        m2 = aType(reshape([0.5, 0.0, 0.7, 0.8], 1, :, 1))
        res, _ = encoder(m2, ps, st)

        @test Array(vec(reduce(vcat, res))) ≈ [0.5, 0.5, 1.2, 2.0]
    end
end

@testitem "Bidirectional" setup = [SharedTestSetup, RecurrentLayersSetup] tags = [
    :recurrent_layers
] begin
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

@testitem "Issue #1305" setup = [SharedTestSetup, RecurrentLayersSetup] tags = [
    :recurrent_layers
] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Chain(
            Recurrence(LSTMCell(4 => 4); return_sequence=true),
            WrappedFunction(x -> Tuple(x)),
        )
        ps, st = Lux.setup(rng, model) |> dev

        x =
            (
                randn(rng, Float32, 4, 10),
                randn(rng, Float32, 4, 10),
                randn(rng, Float32, 4, 10),
            ) |> dev

        _f = (model, x, ps, st) -> begin
            y, _ = model(x, ps, st)
            return sum(sum, y)
        end

        @test_gradients(
            _f,
            model,
            x,
            ps,
            st;
            atol=1.0e-3,
            rtol=1.0e-3,
            # skip_backends=[AutoEnzyme()],
            soft_fail=[AutoFiniteDiff()]
        )
    end
end

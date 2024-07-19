@testitem "RNNCell" setup=[SharedTestSetup] tags=[:recurrent_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        for rnncell in (RNNCell(3 => 5, identity), RNNCell(3 => 5, tanh),
            RNNCell(3 => 5, tanh; use_bias=false),
            RNNCell(3 => 5, identity; use_bias=false),
            RNNCell(3 => 5, identity; use_bias=false, train_state=false))
            display(rnncell)
            ps, st = Lux.setup(rng, rnncell) .|> device
            for x_size in ((3, 2), (3,))
                x = randn(rng, Float32, x_size...) |> aType
                (y, carry), st_ = Lux.apply(rnncell, x, ps, st)

                @jet rnncell(x, ps, st)
                @jet rnncell((x, carry), ps, st)

                function loss_loop_rnncell(p)
                    (y, carry), st_ = rnncell(x, p, st)
                    for i in 1:10
                        (y, carry), st_ = rnncell((x, carry), p, st_)
                    end
                    return sum(abs2, y)
                end

                @test_throws ErrorException ps.train_state

                @eval @test_gradients $loss_loop_rnncell $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
            end
        end

        @testset "Trainable hidden states" begin
            for rnncell in (RNNCell(3 => 5, identity; use_bias=false, train_state=true),
                RNNCell(3 => 5, identity; use_bias=true, train_state=true))
                rnn_no_trainable_state = RNNCell(
                    3 => 5, identity; use_bias=false, train_state=false)
                _ps, _st = Lux.setup(rng, rnn_no_trainable_state) .|> device

                rnncell = RNNCell(3 => 5, identity; use_bias=false, train_state=true)
                ps, st = Lux.setup(rng, rnncell) .|> device
                ps = merge(_ps, (hidden_state=ps.hidden_state,))

                for x_size in ((3, 2), (3,))
                    x = randn(rng, Float32, x_size...) |> aType
                    (_y, _carry), _ = Lux.apply(rnn_no_trainable_state, x, _ps, _st)

                    (y, carry), _ = Lux.apply(rnncell, x, ps, st)
                    @test carry == _carry

                    l, back = Zygote.pullback(
                        p -> sum(abs2, 0 .- rnncell(x, p, st)[1][1]), ps)
                    gs = back(one(l))[1]
                    @test !isnothing(gs.hidden_state)
                end
            end
        end
    end
end

@testitem "LSTMCell" setup=[SharedTestSetup] tags=[:recurrent_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        for lstmcell in (LSTMCell(3 => 5), LSTMCell(3 => 5; use_bias=true),
            LSTMCell(3 => 5; use_bias=false))
            display(lstmcell)
            ps, st = Lux.setup(rng, lstmcell) .|> device

            for x_size in ((3, 2), (3,))
                x = randn(rng, Float32, x_size...) |> aType
                (y, carry), st_ = Lux.apply(lstmcell, x, ps, st)

                @jet lstmcell(x, ps, st)
                @jet lstmcell((x, carry), ps, st)

                function loss_loop_lstmcell(p)
                    (y, carry), st_ = lstmcell(x, p, st)
                    for i in 1:10
                        (y, carry), st_ = lstmcell((x, carry), p, st_)
                    end
                    return sum(abs2, y)
                end

                @eval @test_gradients $loss_loop_lstmcell $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

                @test_throws ErrorException ps.train_state
                @test_throws ErrorException ps.train_memory
            end
        end

        @testset "Trainable hidden states" begin
            for x_size in ((3, 2), (3,))
                x = randn(rng, Float32, x_size...) |> aType
                _lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=false, train_memory=false)
                _ps, _st = Lux.setup(rng, _lstm) .|> device
                (_y, _carry), _ = Lux.apply(_lstm, x, _ps, _st)

                lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=false, train_memory=false)
                ps, st = Lux.setup(rng, lstm) .|> device
                ps = _ps
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(
                    p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test_throws ErrorException gs.bias
                @test_throws ErrorException gs.hidden_state
                @test_throws ErrorException gs.memory

                lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=true, train_memory=false)
                ps, st = Lux.setup(rng, lstm) .|> device
                ps = merge(_ps, (hidden_state=ps.hidden_state,))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(
                    p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test_throws ErrorException gs.bias
                @test !isnothing(gs.hidden_state)
                @test_throws ErrorException gs.memory

                lstm = LSTMCell(
                    3 => 5; use_bias=false, train_state=false, train_memory=true)
                ps, st = Lux.setup(rng, lstm) .|> device
                ps = merge(_ps, (memory=ps.memory,))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(
                    p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test_throws ErrorException gs.bias
                @test_throws ErrorException gs.hidden_state
                @test !isnothing(gs.memory)

                lstm = LSTMCell(3 => 5; use_bias=false, train_state=true, train_memory=true)
                ps, st = Lux.setup(rng, lstm) .|> device
                ps = merge(_ps, (hidden_state=ps.hidden_state, memory=ps.memory))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(
                    p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test_throws ErrorException gs.bias
                @test !isnothing(gs.hidden_state)
                @test !isnothing(gs.memory)

                lstm = LSTMCell(3 => 5; use_bias=true, train_state=true, train_memory=true)
                ps, st = Lux.setup(rng, lstm) .|> device
                ps = merge(
                    _ps, (bias=ps.bias, hidden_state=ps.hidden_state, memory=ps.memory))
                (y, carry), _ = Lux.apply(lstm, x, ps, st)
                l, back = Zygote.pullback(
                    p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test !isnothing(gs.bias)
                @test !isnothing(gs.hidden_state)
                @test !isnothing(gs.memory)
            end
        end
    end
end

@testitem "GRUCell" setup=[SharedTestSetup] tags=[:recurrent_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        for grucell in (GRUCell(3 => 5), GRUCell(3 => 5; use_bias=true),
            GRUCell(3 => 5; use_bias=false))
            display(grucell)
            ps, st = Lux.setup(rng, grucell) .|> device

            for x_size in ((3, 2), (3,))
                x = randn(rng, Float32, x_size...) |> aType
                (y, carry), st_ = Lux.apply(grucell, x, ps, st)

                @jet grucell(x, ps, st)
                @jet grucell((x, carry), ps, st)

                function loss_loop_grucell(p)
                    (y, carry), st_ = grucell(x, p, st)
                    for i in 1:10
                        (y, carry), st_ = grucell((x, carry), p, st_)
                    end
                    return sum(abs2, y)
                end

                @eval @test_gradients $loss_loop_grucell $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu

                @test_throws ErrorException ps.train_state
            end
        end

        @testset "Trainable hidden states" begin
            for x_size in ((3, 2), (3,))
                x = randn(rng, Float32, x_size...) |> aType
                _gru = GRUCell(3 => 5; use_bias=false, train_state=false)
                _ps, _st = Lux.setup(rng, _gru) .|> device
                (_y, _carry), _ = Lux.apply(_gru, x, _ps, _st)

                gru = GRUCell(3 => 5; use_bias=false, train_state=false)
                ps, st = Lux.setup(rng, gru) .|> device
                ps = _ps
                (y, carry), _ = Lux.apply(gru, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test_throws ErrorException gs.bias
                @test_throws ErrorException gs.hidden_state

                gru = GRUCell(3 => 5; use_bias=false, train_state=true)
                ps, st = Lux.setup(rng, gru) .|> device
                ps = merge(_ps, (hidden_state=ps.hidden_state,))
                (y, carry), _ = Lux.apply(gru, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test !isnothing(gs.hidden_state)

                gru = GRUCell(3 => 5; use_bias=true, train_state=true)
                ps, st = Lux.setup(rng, gru) .|> device
                ps = merge(
                    _ps, (bias_h=ps.bias_h, bias_i=ps.bias_i, hidden_state=ps.hidden_state))
                (y, carry), _ = Lux.apply(gru, x, ps, st)
                @test carry == _carry
                l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
                gs = back(one(l))[1]
                @test !isnothing(gs.hidden_state)
            end
        end
    end
end

@testitem "StatefulRecurrentCell" setup=[SharedTestSetup] tags=[:recurrent_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        for _cell in (RNNCell, LSTMCell, GRUCell),
            use_bias in (true, false),
            train_state in (true, false)

            cell = _cell(3 => 5; use_bias, train_state)
            rnn = StatefulRecurrentCell(cell)
            display(rnn)

            for x_size in ((3, 2), (3,))
                x = randn(rng, Float32, x_size...) |> aType
                ps, st = Lux.setup(rng, rnn) .|> device

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

                function loss_loop_rnn(p)
                    y, st_ = rnn(x, p, st)
                    for i in 1:10
                        y, st_ = rnn(x, p, st_)
                    end
                    return sum(abs2, y)
                end

                @eval @test_gradients $loss_loop_rnn $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu
            end
        end
    end
end

@testitem "Recurrence" timeout=3000 setup=[SharedTestSetup] tags=[:recurrent_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        @testset "ordering: $ordering" for ordering in (BatchLastIndex(), TimeLastIndex())
            @testset "cell: $_cell" for _cell in (RNNCell, LSTMCell, GRUCell)
                @testset "use_bias: $use_bias, train_state: $train_state" for use_bias in (
                        true, false),
                    train_state in (true, false)

                    cell = _cell(3 => 5; use_bias, train_state)
                    rnn = Recurrence(cell; ordering)
                    rnn_seq = Recurrence(cell; ordering, return_sequence=true)
                    display(rnn)

                    # Batched Time Series
                    @testset "typeof(x): $(typeof(x))" for x in (
                        randn(rng, Float32, 3, 4, 2) |> aType,
                        Tuple(randn(rng, Float32, 3, 2) for _ in 1:4) .|> aType,
                        [randn(rng, Float32, 3, 2) for _ in 1:4] .|> aType)
                        # Fix data ordering for testing
                        if ordering isa TimeLastIndex && x isa AbstractArray && ndims(x) ≥ 2
                            x = permutedims(x,
                                (ntuple(identity, ndims(x) - 2)..., ndims(x), ndims(x) - 1))
                        end

                        ps, st = Lux.setup(rng, rnn) .|> device
                        y, st_ = rnn(x, ps, st)
                        y_, st__ = rnn_seq(x, ps, st)

                        @jet rnn(x, ps, st)
                        @jet rnn_seq(x, ps, st)

                        @test size(y) == (5, 2)
                        @test length(y_) == 4
                        @test all(x -> size(x) == (5, 2), y_)

                        __f = p -> sum(first(rnn(x, p, st)))
                        @eval @test_gradients $__f $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu

                        __f = p -> sum(Base.Fix1(sum, abs2), first(rnn_seq(x, p, st)))
                        @eval @test_gradients $__f $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu
                    end

                    # Batched Time Series without data batches
                    @testset "typeof(x): $(typeof(x))" for x in (
                        randn(rng, Float32, 3, 4) |> aType,
                        Tuple(randn(rng, Float32, 3) for _ in 1:4) .|> aType,
                        [randn(rng, Float32, 3) for _ in 1:4] .|> aType)
                        ps, st = Lux.setup(rng, rnn) .|> device
                        y, st_ = rnn(x, ps, st)
                        y_, st__ = rnn_seq(x, ps, st)

                        @jet rnn(x, ps, st)
                        @jet rnn_seq(x, ps, st)

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

                        __f = p -> sum(first(rnn(x, p, st)))
                        @eval @test_gradients $__f $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu

                        __f = p -> sum(Base.Fix1(sum, abs2), first(rnn_seq(x, p, st)))
                        @eval @test_gradients $__f $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu
                    end
                end
            end
        end

        # Ordering Check: https://github.com/LuxDL/Lux.jl/issues/302
        encoder = Recurrence(
            RNNCell(1 => 1, identity;
                init_weight=(rng, args...; kwargs...) -> ones(args...; kwargs...),
                init_state=(rng, args...; kwargs...) -> zeros(args...; kwargs...),
                init_bias=(rng, args...; kwargs...) -> zeros(args...; kwargs...));
            return_sequence=true)
        ps, st = Lux.setup(rng, encoder) .|> device
        m2 = reshape([0.5, 0.0, 0.7, 0.8], 1, :, 1) |> aType
        res, _ = encoder(m2, ps, st)

        @test Array(vec(reduce(vcat, res))) ≈ [0.5, 0.5, 1.2, 2.0]
    end
end

@testitem "Bidirectional" timeout=3000 setup=[SharedTestSetup] tags=[:recurrent_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        @testset "cell: $_cell" for _cell in (RNNCell, LSTMCell, GRUCell)
            cell = _cell(3 => 5)
            bi_rnn = BidirectionalRNN(cell)
            bi_rnn_no_merge = BidirectionalRNN(cell; merge_mode=nothing)
            display(bi_rnn)

            # Batched Time Series
            x = randn(rng, Float32, 3, 4, 2) |> aType
            ps, st = Lux.setup(rng, bi_rnn) .|> device
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

            __f = p -> sum(Base.Fix1(sum, abs2), first(bi_rnn(x, p, st)))
            @eval @test_gradients $__f $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu

            __f = p -> begin
                (y1, y2), st_ = bi_rnn_no_merge(x, p, st)
                return sum(Base.Fix1(sum, abs2), y1) + sum(Base.Fix1(sum, abs2), y2)
            end
            @eval @test_gradients $__f $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu

            @testset "backward_cell: $_backward_cell" for _backward_cell in (
                RNNCell, LSTMCell, GRUCell)
                cell = _cell(3 => 5)
                backward_cell = _backward_cell(3 => 5)
                bi_rnn = BidirectionalRNN(cell, backward_cell)
                bi_rnn_no_merge = BidirectionalRNN(cell, backward_cell; merge_mode=nothing)
                display(bi_rnn)

                # Batched Time Series
                x = randn(rng, Float32, 3, 4, 2) |> aType
                ps, st = Lux.setup(rng, bi_rnn) .|> device
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

                __f = p -> sum(Base.Fix1(sum, abs2), first(bi_rnn(x, p, st)))
                @eval @test_gradients $__f $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu

                __f = p -> begin
                    (y1, y2), st_ = bi_rnn_no_merge(x, p, st)
                    return sum(Base.Fix1(sum, abs2), y1) + sum(Base.Fix1(sum, abs2), y2)
                end
                @eval @test_gradients $__f $ps atol=1e-2 rtol=1e-2 gpu_testing=$ongpu
            end
        end
    end
end

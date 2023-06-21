using Lux, Test

include("../test_utils.jl")

rng = get_stable_rng(12345)

@testset "$mode: RNNCell" for (mode, aType, device, ongpu) in MODES
    for rnncell in (RNNCell(3 => 5, identity),
        RNNCell(3 => 5, tanh),
        RNNCell(3 => 5, tanh; use_bias=false),
        RNNCell(3 => 5, identity; use_bias=false),
        RNNCell(3 => 5, identity; use_bias=false, train_state=false))
        display(rnncell)
        ps, st = Lux.setup(rng, rnncell) .|> device
        x = randn(rng, Float32, 3, 2) |> aType
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

    @testset "Trainable hidden states" begin
        for rnncell in (RNNCell(3 => 5, identity; use_bias=false, train_state=true),
            RNNCell(3 => 5, identity; use_bias=true, train_state=true))
            rnn_no_trainable_state = RNNCell(3 => 5,
                identity;
                use_bias=false,
                train_state=false)
            x = randn(rng, Float32, 3, 2) |> aType
            _ps, _st = Lux.setup(rng, rnn_no_trainable_state) .|> device
            (_y, _carry), _ = Lux.apply(rnn_no_trainable_state, x, _ps, _st)

            rnncell = RNNCell(3 => 5, identity; use_bias=false, train_state=true)
            ps, st = Lux.setup(rng, rnncell) .|> device
            ps = merge(_ps, (hidden_state=ps.hidden_state,))
            (y, carry), _ = Lux.apply(rnncell, x, ps, st)
            @test carry == _carry

            l, back = Zygote.pullback(p -> sum(abs2, 0 .- rnncell(x, p, st)[1][1]), ps)
            gs = back(one(l))[1]
            @test !isnothing(gs.hidden_state)
        end
    end
end

@testset "$mode: LSTMCell" for (mode, aType, device, ongpu) in MODES
    for lstmcell in (LSTMCell(3 => 5),
        LSTMCell(3 => 5; use_bias=true),
        LSTMCell(3 => 5; use_bias=false))
        display(lstmcell)
        ps, st = Lux.setup(rng, lstmcell) .|> device
        x = randn(rng, Float32, 3, 2) |> aType
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

    @testset "Trainable hidden states" begin
        x = randn(rng, Float32, 3, 2) |> aType
        _lstm = LSTMCell(3 => 5; use_bias=false, train_state=false, train_memory=false)
        _ps, _st = Lux.setup(rng, _lstm) .|> device
        (_y, _carry), _ = Lux.apply(_lstm, x, _ps, _st)

        lstm = LSTMCell(3 => 5; use_bias=false, train_state=false, train_memory=false)
        ps, st = Lux.setup(rng, lstm) .|> device
        ps = _ps
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test_throws ErrorException gs.bias
        @test_throws ErrorException gs.hidden_state
        @test_throws ErrorException gs.memory

        lstm = LSTMCell(3 => 5; use_bias=false, train_state=true, train_memory=false)
        ps, st = Lux.setup(rng, lstm) .|> device
        ps = merge(_ps, (hidden_state=ps.hidden_state,))
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test_throws ErrorException gs.bias
        @test !isnothing(gs.hidden_state)
        @test_throws ErrorException gs.memory

        lstm = LSTMCell(3 => 5; use_bias=false, train_state=false, train_memory=true)
        ps, st = Lux.setup(rng, lstm) .|> device
        ps = merge(_ps, (memory=ps.memory,))
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test_throws ErrorException gs.bias
        @test_throws ErrorException gs.hidden_state
        @test !isnothing(gs.memory)

        lstm = LSTMCell(3 => 5; use_bias=false, train_state=true, train_memory=true)
        ps, st = Lux.setup(rng, lstm) .|> device
        ps = merge(_ps, (hidden_state=ps.hidden_state, memory=ps.memory))
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test_throws ErrorException gs.bias
        @test !isnothing(gs.hidden_state)
        @test !isnothing(gs.memory)

        lstm = LSTMCell(3 => 5; use_bias=true, train_state=true, train_memory=true)
        ps, st = Lux.setup(rng, lstm) .|> device
        ps = merge(_ps, (bias=ps.bias, hidden_state=ps.hidden_state, memory=ps.memory))
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test !isnothing(gs.bias)
        @test !isnothing(gs.hidden_state)
        @test !isnothing(gs.memory)
    end
end

@testset "$mode: GRUCell" for (mode, aType, device, ongpu) in MODES
    for grucell in (GRUCell(3 => 5),
        GRUCell(3 => 5; use_bias=true),
        GRUCell(3 => 5; use_bias=false))
        display(grucell)
        ps, st = Lux.setup(rng, grucell) .|> device
        x = randn(rng, Float32, 3, 2) |> aType
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

    @testset "Trainable hidden states" begin
        x = randn(rng, Float32, 3, 2) |> aType
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
        ps = merge(_ps, (bias_h=ps.bias_h, bias_i=ps.bias_i, hidden_state=ps.hidden_state))
        (y, carry), _ = Lux.apply(gru, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test !isnothing(gs.hidden_state)
    end
end

@testset "$mode: StatefulRecurrentCell" for (mode, aType, device, ongpu) in MODES
    for _cell in (RNNCell, LSTMCell, GRUCell),
        use_bias in (true, false),
        train_state in (true, false)

        cell = _cell(3 => 5; use_bias, train_state)
        rnn = StatefulRecurrentCell(cell)
        display(rnn)
        x = randn(rng, Float32, 3, 2) |> aType
        ps, st = Lux.setup(rng, rnn) .|> device

        y, st_ = rnn(x, ps, st)

        @jet rnn(x, ps, st)
        @jet rnn(x, ps, st_)

        @test size(y) == (5, 2)
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

@testset "$mode: Recurrence" for (mode, aType, device, ongpu) in MODES
    for _cell in (RNNCell, LSTMCell, GRUCell),
        use_bias in (true, false),
        train_state in (true, false)

        cell = _cell(3 => 5; use_bias, train_state)
        rnn = Recurrence(cell)
        rnn_seq = Recurrence(cell; return_sequence=true)
        display(rnn)

        # Batched Time Series
        for x in (randn(rng, Float32, 3, 4, 2) |> aType,
            Tuple(randn(rng, Float32, 3, 2) for _ in 1:4) .|> aType,
            [randn(rng, Float32, 3, 2) for _ in 1:4] .|> aType)
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
    end

    # Ordering Check: https://github.com/LuxDL/Lux.jl/issues/302
    encoder = Recurrence(RNNCell(1 => 1,
            identity;
            init_weight=ones,
            init_state=zeros,
            init_bias=zeros);
        return_sequence=true)
    ps, st = Lux.setup(rng, encoder) .|> device
    m2 = reshape([0.5, 0.0, 0.7, 0.8], 1, :, 1) |> aType
    res, _ = encoder(m2, ps, st)

    @test Array(vec(reduce(vcat, res))) ≈ [0.5, 0.5, 1.2, 2.0]
end

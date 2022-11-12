using Lux, NNlib, Random, Test

include("../test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "RNNCell" begin
    for rnncell in (RNNCell(3 => 5, identity), RNNCell(3 => 5, tanh),
                    RNNCell(3 => 5, tanh; use_bias=false),
                    RNNCell(3 => 5, identity; use_bias=false),
                    RNNCell(3 => 5, identity; use_bias=false, train_state=false))
        display(rnncell)
        ps, st = Lux.setup(rng, rnncell)
        x = randn(rng, Float32, 3, 2)
        (y, carry), st_ = Lux.apply(rnncell, x, ps, st)

        run_JET_tests(rnncell, x, ps, st)
        run_JET_tests(rnncell, (x, carry), ps, st_)

        function loss_loop_rnncell(p)
            (y, carry), st_ = rnncell(x, p, st)
            for i in 1:10
                (y, carry), st_ = rnncell((x, carry), p, st_)
            end
            return sum(abs2, y)
        end

        @test_throws ErrorException ps.train_state

        test_gradient_correctness_fdm(loss_loop_rnncell, ps; atol=1e-3, rtol=1e-3)
    end

    @testset "Trainable hidden states" begin for rnncell in (RNNCell(3 => 5, identity;
                                                                     use_bias=false,
                                                                     train_state=true),
                                                             RNNCell(3 => 5, identity;
                                                                     use_bias=true,
                                                                     train_state=true))
        rnn_no_trainable_state = RNNCell(3 => 5, identity; use_bias=false,
                                         train_state=false)
        x = randn(rng, Float32, 3, 2)
        _ps, _st = Lux.setup(rng, rnn_no_trainable_state)
        (_y, _carry), _ = Lux.apply(rnn_no_trainable_state, x, _ps, _st)

        rnncell = RNNCell(3 => 5, identity; use_bias=false, train_state=true)
        ps, st = Lux.setup(rng, rnncell)
        ps = merge(_ps, (hidden_state=ps.hidden_state,))
        (y, carry), _ = Lux.apply(rnncell, x, ps, st)
        @test carry == _carry

        l, back = Zygote.pullback(p -> sum(abs2, 0 .- rnncell(x, p, st)[1][1]), ps)
        gs = back(one(l))[1]
        @test !isnothing(gs.hidden_state)
    end end

    # Deprecated Functionality (Remove in v0.5)
    @testset "Deprecations" begin
        @test_deprecated RNNCell(3 => 5, relu; bias=false)
        @test_deprecated RNNCell(3 => 5, relu; bias=true)
        @test_throws ArgumentError RNNCell(3 => 5, relu; bias=false, use_bias=false)
    end
end

@testset "LSTMCell" begin
    for lstmcell in (LSTMCell(3 => 5), LSTMCell(3 => 5; use_bias=true),
                     LSTMCell(3 => 5; use_bias=false))
        display(lstmcell)
        ps, st = Lux.setup(rng, lstmcell)
        x = randn(rng, Float32, 3, 2)
        (y, carry), st_ = Lux.apply(lstmcell, x, ps, st)

        run_JET_tests(lstmcell, x, ps, st)
        run_JET_tests(lstmcell, (x, carry), ps, st_)

        function loss_loop_lstmcell(p)
            (y, carry), st_ = lstmcell(x, p, st)
            for i in 1:10
                (y, carry), st_ = lstmcell((x, carry), p, st_)
            end
            return sum(abs2, y)
        end

        test_gradient_correctness_fdm(loss_loop_lstmcell, ps; atol=1e-3, rtol=1e-3)

        @test_throws ErrorException ps.train_state
        @test_throws ErrorException ps.train_memory
    end

    @testset "Trainable hidden states" begin
        x = randn(rng, Float32, 3, 2)
        _lstm = LSTMCell(3 => 5; use_bias=false, train_state=false, train_memory=false)
        _ps, _st = Lux.setup(rng, _lstm)
        (_y, _carry), _ = Lux.apply(_lstm, x, _ps, _st)

        lstm = LSTMCell(3 => 5; use_bias=false, train_state=false, train_memory=false)
        ps, st = Lux.setup(rng, lstm)
        ps = _ps
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test_throws ErrorException gs.bias
        @test_throws ErrorException gs.hidden_state
        @test_throws ErrorException gs.memory

        lstm = LSTMCell(3 => 5; use_bias=false, train_state=true, train_memory=false)
        ps, st = Lux.setup(rng, lstm)
        ps = merge(_ps, (hidden_state=ps.hidden_state,))
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test_throws ErrorException gs.bias
        @test !isnothing(gs.hidden_state)
        @test_throws ErrorException gs.memory

        lstm = LSTMCell(3 => 5; use_bias=false, train_state=false, train_memory=true)
        ps, st = Lux.setup(rng, lstm)
        ps = merge(_ps, (memory=ps.memory,))
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test_throws ErrorException gs.bias
        @test_throws ErrorException gs.hidden_state
        @test !isnothing(gs.memory)

        lstm = LSTMCell(3 => 5; use_bias=false, train_state=true, train_memory=true)
        ps, st = Lux.setup(rng, lstm)
        ps = merge(_ps, (hidden_state=ps.hidden_state, memory=ps.memory))
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test_throws ErrorException gs.bias
        @test !isnothing(gs.hidden_state)
        @test !isnothing(gs.memory)

        lstm = LSTMCell(3 => 5; use_bias=true, train_state=true, train_memory=true)
        ps, st = Lux.setup(rng, lstm)
        ps = merge(_ps, (bias=ps.bias, hidden_state=ps.hidden_state, memory=ps.memory))
        (y, carry), _ = Lux.apply(lstm, x, ps, st)
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(lstm(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test !isnothing(gs.bias)
        @test !isnothing(gs.hidden_state)
        @test !isnothing(gs.memory)
    end
end

@testset "GRUCell" begin
    for grucell in (GRUCell(3 => 5), GRUCell(3 => 5; use_bias=true),
                    GRUCell(3 => 5; use_bias=false))
        display(grucell)
        ps, st = Lux.setup(rng, grucell)
        x = randn(rng, Float32, 3, 2)
        (y, carry), st_ = Lux.apply(grucell, x, ps, st)

        run_JET_tests(grucell, x, ps, st)
        run_JET_tests(grucell, (x, carry), ps, st_)

        function loss_loop_grucell(p)
            (y, carry), st_ = grucell(x, p, st)
            for i in 1:10
                (y, carry), st_ = grucell((x, carry), p, st_)
            end
            return sum(abs2, y)
        end

        test_gradient_correctness_fdm(loss_loop_grucell, ps; atol=1e-3, rtol=1e-3)

        @test_throws ErrorException ps.train_state
    end

    @testset "Trainable hidden states" begin
        x = randn(rng, Float32, 3, 2)
        _gru = GRUCell(3 => 5; use_bias=false, train_state=false)
        _ps, _st = Lux.setup(rng, _gru)
        (_y, _carry), _ = Lux.apply(_gru, x, _ps, _st)

        gru = GRUCell(3 => 5; use_bias=false, train_state=false)
        ps, st = Lux.setup(rng, gru)
        ps = _ps
        (y, carry), _ = Lux.apply(gru, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test_throws ErrorException gs.bias
        @test_throws ErrorException gs.hidden_state

        gru = GRUCell(3 => 5; use_bias=false, train_state=true)
        ps, st = Lux.setup(rng, gru)
        ps = merge(_ps, (hidden_state=ps.hidden_state,))
        (y, carry), _ = Lux.apply(gru, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test !isnothing(gs.hidden_state)

        gru = GRUCell(3 => 5; use_bias=true, train_state=true)
        ps, st = Lux.setup(rng, gru)
        ps = merge(_ps, (bias_h=ps.bias_h, bias_i=ps.bias_i, hidden_state=ps.hidden_state))
        (y, carry), _ = Lux.apply(gru, x, ps, st)
        @test carry == _carry
        l, back = Zygote.pullback(p -> sum(abs2, 0 .- sum(gru(x, p, st)[1][1])), ps)
        gs = back(one(l))[1]
        @test !isnothing(gs.hidden_state)
    end
end

@testset "StatefulRecurrentCell" begin for _cell in (RNNCell, LSTMCell, GRUCell),
                                           use_bias in (true, false),
                                           train_state in (true, false)

    cell = _cell(3 => 5; use_bias, train_state)
    rnn = StatefulRecurrentCell(cell)
    display(rnn)
    x = randn(rng, Float32, 3, 2)
    ps, st = Lux.setup(rng, rnn)

    y, st_ = rnn(x, ps, st)

    run_JET_tests(rnn, x, ps, st)
    run_JET_tests(rnn, x, ps, st_)

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

    test_gradient_correctness_fdm(loss_loop_rnn, ps; atol=1e-3, rtol=1e-3)
end end

@testset "Recurrence" begin for _cell in (RNNCell, LSTMCell, GRUCell),
                                use_bias in (true, false),
                                train_state in (true, false)

    cell = _cell(3 => 5; use_bias, train_state)
    rnn = Recurrence(cell)
    display(rnn)

    # Batched Time Series
    x = randn(rng, Float32, 3, 4, 2)
    ps, st = Lux.setup(rng, rnn)
    y, st_ = rnn(x, ps, st)

    run_JET_tests(rnn, x, ps, st)

    @test size(y) == (5, 2)

    test_gradient_correctness_fdm(p -> sum(rnn(x, p, st)[1]), ps; atol=1e-3, rtol=1e-3)

    # Tuple of Time Series
    x = Tuple(randn(rng, Float32, 3, 2) for _ in 1:4)
    ps, st = Lux.setup(rng, rnn)
    y, st_ = rnn(x, ps, st)

    run_JET_tests(rnn, x, ps, st)

    @test size(y) == (5, 2)

    test_gradient_correctness_fdm(p -> sum(rnn(x, p, st)[1]), ps; atol=1e-3, rtol=1e-3)

    # Vector of Time Series
    x = [randn(rng, Float32, 3, 2) for _ in 1:4]
    ps, st = Lux.setup(rng, rnn)
    y, st_ = rnn(x, ps, st)

    run_JET_tests(rnn, x, ps, st)

    @test size(y) == (5, 2)

    test_gradient_correctness_fdm(p -> sum(rnn(x, p, st)[1]), ps; atol=1e-3, rtol=1e-3)
end end

@testset "multigate" begin
    x = rand(6, 5)
    res, (dx,) = Zygote.withgradient(x) do x
        x1, _, x3 = Lux.multigate(x, Val(3))
        return sum(x1) + sum(x3 .* 2)
    end
    @test res == sum(x[1:2, :]) + 2sum(x[5:6, :])
    @test dx == [ones(2, 5); zeros(2, 5); fill(2, 2, 5)]
end

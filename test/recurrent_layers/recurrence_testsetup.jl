# Helper function for recurrence layer tests
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

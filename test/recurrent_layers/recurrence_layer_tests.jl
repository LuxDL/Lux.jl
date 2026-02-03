using LuxTestUtils: check_approx

include("../shared_testsetup.jl")

# Helper functions from RecurrenceTestSetup
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

const RECURRENCE_ALL_TEST_CONFIGS = Iterators.product(
    (BatchLastIndex(), TimeLastIndex()),
    (RNNCell, LSTMCell, GRUCell),
    (true, false),
    (true, false),
)

const RECURRENCE_TEST_BLOCKS = collect(
    Iterators.partition(
        RECURRENCE_ALL_TEST_CONFIGS, ceil(Int, length(RECURRENCE_ALL_TEST_CONFIGS) / 2)
    ),
)

@testset "Recurrence: Group 1" begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset for (ordering, cell, use_bias, train_state) in RECURRENCE_TEST_BLOCKS[1]
            test_recurrence_layer(
                mode, aType, dev, ongpu, ordering, cell, use_bias, train_state
            )
        end
    end
end

@testset "Recurrence: Group 2" begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset for (ordering, cell, use_bias, train_state) in RECURRENCE_TEST_BLOCKS[2]
            test_recurrence_layer(
                mode, aType, dev, ongpu, ordering, cell, use_bias, train_state
            )
        end
    end
end

@testset "Recurrence Ordering Check #302" begin
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

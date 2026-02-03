using LuxTestUtils: check_approx

include("../shared_testsetup.jl")

@testset "Issue #1305" begin
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

using LuxTestUtils: check_approx

include("../shared_testsetup.jl")

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

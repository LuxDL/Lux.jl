using ForwardDiff, LuxLib, Test, StableRNGs
using LuxTestUtils: check_approx

include("../shared_testsetup.jl")

@testset "ForwardDiff dropout" begin
    rng = StableRNG(12345)

    @testset "$mode: dropout" for (mode, aType, ongpu, fp64) in MODES
        x = aType(randn(rng, Float32, 10, 2))
        x_dual = ForwardDiff.Dual.(x)

        @test_nowarn dropout(rng, x_dual, 0.5f0, Val(true), 2.0f0, :)

        x_dropout = dropout(rng, x, 0.5f0, Val(true), 2.0f0, :)[1]
        x_dual_dropout =
            ForwardDiff.value.(dropout(rng, x_dual, 0.5f0, Val(true), 2.0f0, :)[1])

        @test check_approx(x_dropout, x_dual_dropout)
    end
end

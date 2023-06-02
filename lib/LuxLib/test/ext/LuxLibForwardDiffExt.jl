using LuxLib, ForwardDiff, Test

include("../test_utils.jl")

rng = get_stable_rng(12345)

@testset "dropout" begin
    if cpu_testing()
        x = randn(rng, Float32, 10, 2)
        x_dual = ForwardDiff.Dual.(x)

        @test_nowarn dropout(rng, x_dual, 0.5f0, Val(true); dims=:)

        x_dropout = dropout(rng, x, 0.5f0, Val(true); dims=:)[1]
        x_dual_dropout = ForwardDiff.value.(dropout(rng, x_dual, 0.5f0, Val(true); dims=:)[1])

        @test check_approx(x_dropout, x_dual_dropout)
    end
end

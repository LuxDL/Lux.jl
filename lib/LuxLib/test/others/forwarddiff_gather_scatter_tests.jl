using ForwardDiff, NNlib, LuxLib, Test, StableRNGs
using LuxTestUtils: check_approx

@testset "ForwardDiff gather/scatter" begin
    rng = StableRNG(12345)

    @testset "gather" begin
        a = [1, 20, 300, 4000]
        ∂a = [-1, -20, -300, -4000]
        a_dual = ForwardDiff.Dual.(a, ∂a)

        res = NNlib.gather(a_dual, [2, 4, 2])
        @test ForwardDiff.value.(res) == [20, 4000, 20]
        @test ForwardDiff.partials.(res, 1) == [-20, -4000, -20]

        a = [1 2 3; 4 5 6]
        ∂a = [-1 -2 -3; -4 -5 -6]
        a_dual = ForwardDiff.Dual.(a, ∂a)

        res = NNlib.gather(a_dual, [1, 3, 1, 3, 1])
        @test ForwardDiff.value.(res) == [1 3 1 3 1; 4 6 4 6 4]
        @test ForwardDiff.partials.(res, 1) == [-1 -3 -1 -3 -1; -4 -6 -4 -6 -4]
    end

    @testset "scatter" begin
        a = [10, 100, 1000]
        ∂a = [-10, -100, -1000]
        a_dual = ForwardDiff.Dual.(a, ∂a)

        res = NNlib.scatter(+, a_dual, [3, 1, 2])
        @test ForwardDiff.value.(res) == [100, 1000, 10]
        @test ForwardDiff.partials.(res, 1) == [-100, -1000, -10]

        a = [1 2 3 4; 5 6 7 8]
        ∂a = [-1 -2 -3 -4; -5 -6 -7 -8]
        a_dual = ForwardDiff.Dual.(a, ∂a)

        res = NNlib.scatter(+, a_dual, [2, 1, 1, 5])
        @test ForwardDiff.value.(res) == [5 1 0 0 4; 13 5 0 0 8]
        @test ForwardDiff.partials.(res, 1) == [-5 -1 0 0 -4; -13 -5 0 0 -8]

        a = [10, 200, 3000]
        ∂a = [-10, -200, -3000]
        a_dual = ForwardDiff.Dual.(a, ∂a)

        res = NNlib.scatter(*, a_dual, [1, 4, 2]; init=10, dstsize=6)
        @test ForwardDiff.value.(res) == [100, 30000, 10, 2000, 10, 10]
        @test ForwardDiff.partials.(res, 1) == [-100, -30000, 10, -2000, 10, 10]
    end
end

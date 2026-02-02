include("bmm_testsetup.jl")

using NNlib: batched_adjoint, batched_transpose

@testset "BatchedAdjOrTrans interface" begin
    rng = StableRNG(1234)

    @testset "Float64 × $(TB)" for TB in [Float64, Float32]
        A = randn(rng, 7, 5, 3)
        B = randn(rng, TB, 5, 7, 3)
        C = randn(rng, 7, 6, 3)

        function interface_tests(X, _X)
            @test length(_X) == length(X)
            @test size(_X) == (size(X, 2), size(X, 1), size(X, 3))
            @test axes(_X) == (axes(X, 2), axes(X, 1), axes(X, 3))

            @test getindex(_X, 2, 3, 3) == getindex(X, 3, 2, 3)
            @test getindex(_X, 5, 4, 1) == getindex(X, 4, 5, 1)

            setindex!(_X, 2.0, 2, 4, 1)
            @test getindex(_X, 2, 4, 1) == 2.0
            setindex!(_X, 3.0, 1, 2, 2)
            @test getindex(_X, 1, 2, 2) == 3.0

            _sim = similar(_X, TB, (2, 3))
            @test size(_sim) == (2, 3)
            @test typeof(_sim) == Array{TB,2}

            _sim = similar(_X, TB)
            @test length(_sim) == length(_X)
            @test typeof(_sim) == Array{TB,3}

            _sim = similar(_X, (2, 3))
            @test size(_sim) == (2, 3)
            @test typeof(_sim) == Array{Float64,2}

            _sim = similar(_X)
            @test length(_sim) == length(_X)
            @test typeof(_sim) == Array{Float64,3}

            @test parent(_X) == _X.parent
        end

        for (X, _X) in zip([A, B, C], map(batched_adjoint, [A, B, C]))
            interface_tests(X, _X)

            @test -_X == NNlib.BatchedAdjoint(-_X.parent)

            _copyX = copy(_X)
            @test _X == _copyX

            setindex!(_copyX, 2.0, 1, 2, 1)
            @test _X != _copyX
        end

        for (X, _X) in zip([A, B, C], map(batched_transpose, [A, B, C]))
            interface_tests(X, _X)

            @test -_X == NNlib.BatchedTranspose(-_X.parent)

            _copyX = copy(_X)
            @test _X == _copyX

            setindex!(_copyX, 2.0, 1, 2, 1)
            @test _X != _copyX
        end
    end
end

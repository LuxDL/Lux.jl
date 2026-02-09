using LuxLib, Test
using LuxLib: batched_matmul
using NNlib: batched_vec, ⊠

include("bmm_testsetup.jl")

@testset "batched_matmul(ndims < 3)" begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        !fp64 && continue

        @testset "Float64 × $(TB)" for TB in [Float64, ComplexF64]
            A = aType(randn(rng, 3, 3, 3))
            M = aType(rand(rng, TB, 3, 3)) .+ im
            V = aType(rand(rng, TB, 3))

            # These are all reshaped and sent to batched_matmul(3-array, 3-array)
            @test batched_matmul(A, M) ≈ cat([A[:, :, k] * M for k in 1:3]...; dims=3)
            @test batched_matmul(A, M') ≈ cat([A[:, :, k] * M' for k in 1:3]...; dims=3)
            @test A ⊠ transpose(M) ≈
                cat([A[:, :, k] * transpose(M) for k in 1:3]...; dims=3)

            @test batched_matmul(M, A) ≈ cat([M * A[:, :, k] for k in 1:3]...; dims=3)
            @test batched_matmul(M', A) ≈ cat([M' * A[:, :, k] for k in 1:3]...; dims=3)
            @test transpose(M) ⊠ A ≈
                cat([transpose(M) * A[:, :, k] for k in 1:3]...; dims=3)

            # batched_vec
            @test batched_vec(A, M) ≈ hcat([A[:, :, k] * M[:, k] for k in 1:3]...)
            @test batched_vec(A, M') ≈ hcat([A[:, :, k] * (M')[:, k] for k in 1:3]...)
            @test batched_vec(A, V) ≈ hcat([A[:, :, k] * V for k in 1:3]...)
        end
    end
end

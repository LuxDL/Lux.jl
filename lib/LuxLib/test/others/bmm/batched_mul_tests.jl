using LuxLib, Test
using LuxLib: batched_matmul
using NNlib: batched_transpose, batched_adjoint, ⊠

include("bmm_testsetup.jl")

@testset "batched_mul" begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "batched_mul: Float64 × $(TB)" for TB in [Float64, Float32]
            !fp64 && continue

            @testset "real" begin
                A = aType(randn(rng, 7, 5, 3))
                B = aType(randn(rng, TB, 5, 7, 3))
                C = aType(randn(rng, 7, 6, 3))

                @test batched_matmul(A, B) ≈ bmm_test(A, B)
                @test batched_matmul(batched_transpose(A), batched_transpose(B)) ≈
                    bmm_test(A, B; transA=true, transB=true)
                @test batched_matmul(batched_transpose(A), C) ≈ bmm_test(A, C; transA=true)
                @test batched_matmul(A, batched_transpose(A)) ≈ bmm_test(A, A; transB=true)
            end

            @testset "complex" begin
                cA = aType(randn(rng, Complex{Float64}, 7, 5, 3))
                cB = aType(randn(rng, Complex{TB}, 5, 7, 3))
                cC = aType(randn(rng, Complex{Float64}, 7, 6, 3))

                @test batched_matmul(cA, cB) ≈ bmm_adjtest(cA, cB)
                @test batched_matmul(batched_adjoint(cA), batched_adjoint(cB)) ≈
                    bmm_adjtest(cA, cB; adjA=true, adjB=true)
                @test batched_matmul(batched_adjoint(cA), cC) ≈
                    bmm_adjtest(cA, cC; adjA=true)
                @test batched_matmul(cA, batched_adjoint(cA)) ≈
                    bmm_adjtest(cA, cA; adjB=true)

                @testset "Integers" begin
                    TBi = TB == Float64 ? Int64 : Int32
                    iA = aType(rand(rng, 1:99, 7, 5, 3))
                    iB = aType(TB.(rand(rng, 1:99, 5, 7, 3)))
                    iC = aType(zeros(Int, 7, 6, 3))

                    @test batched_matmul(iA, iB) == bmm_adjtest(iA, iB)
                    @test batched_matmul(cA, iB) ≈ bmm_adjtest(cA, iB)
                end
            end

            @testset "Errors" begin
                @test_throws DimensionMismatch batched_matmul(
                    aType(rand(rng, 2, 2, 2)), aType(rand(rng, TB, 2, 2, 10))
                )
                @test_throws DimensionMismatch batched_matmul(
                    aType(rand(rng, 2, 2, 2)), aType(rand(rng, TB, 10, 2, 2))
                )
            end

            @testset "PermutedDimsArrays" begin
                if !ongpu
                    for perm in [(1, 3, 2), (2, 1, 3), (3, 2, 1)],
                        fun in [identity, batched_adjoint],
                        ty in [identity, complex]

                        A = aType(randn(rng, ty(Float64), 4, 4, 4))
                        B = aType(randn(rng, ty(TB), 4, 4, 4))

                        @test batched_matmul(fun(A), PermutedDimsArray(B, perm)) ≈
                            batched_matmul(fun(A), permutedims(B, perm))
                        @test batched_matmul(fun(PermutedDimsArray(A, perm)), B) ≈
                            batched_matmul(fun(permutedims(A, perm)), B)
                    end
                end
            end

            @testset "Large output, multi-threaded path" begin
                if TB == Float64
                    N = 50
                    A = aType(rand(rng, N, N, N))
                    B = aType(rand(rng, N, N, N))
                    C = reshape(
                        reduce(hcat, [vec(A[:, :, k] * B[:, :, k]) for k in 1:N]), N, N, N
                    )
                    @test C ≈ A ⊠ B

                    D = aType(rand(rng, N, N, 1))
                    E = reshape(
                        reduce(hcat, [vec(A[:, :, k] * D[:, :, 1]) for k in 1:N]), N, N, N
                    )
                    @test E ≈ A ⊠ D
                end
            end
        end
    end
end

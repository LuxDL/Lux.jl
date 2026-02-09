include("bmm_testsetup.jl")

using LuxLib: batched_matmul
using NNlib: batched_transpose, batched_adjoint

@testset "batched_mul: trivial dimensions & unit strides" begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        !fp64 && continue

        @testset "Float64 × $(TB)" for TB in [Float64]
            @testset "trivial dimensions & unit strides" begin
                @testset "$tA(rand$((sA..., 3))) ⊠ $tB(rand$((sB..., 3)))" for tA in [
                        identity, batched_adjoint, batched_transpose, perm_12, perm_23
                    ],
                    sA in [(1, 1), (1, 3), (3, 1), (3, 3)],
                    tB in [identity, batched_adjoint, batched_transpose, perm_12, perm_23],
                    sB in [(1, 1), (1, 3), (3, 1), (3, 3)]

                    A = aType(tA(rand(rng, TB, sA..., 3)))
                    B = aType(tB(rand(rng, TB, sB..., 3)))

                    if size(A, 2) != size(B, 1) || size(A, 3) != 3 || size(B, 3) != 3
                        @test true # avoid a warning in ReTestItems.jl
                        continue
                    end

                    C = cat(
                        A[:, :, 1] * B[:, :, 1],
                        A[:, :, 2] * B[:, :, 2],
                        A[:, :, 3] * B[:, :, 3];
                        dims=3,
                    )
                    @test batched_matmul(A, B) ≈ C
                end
            end
        end
    end
end

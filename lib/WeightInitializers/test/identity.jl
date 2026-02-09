using LinearAlgebra, WeightInitializers, Test

@testset "Identity Initialization" begin
    @testset "2D identity matrices" begin
        # Square matrix should be identity
        mat = identity_init(5, 5)
        @test mat ≈ Matrix{Float32}(I, 5, 5)
        @test diag(mat) == ones(Float32, 5)
        # Check off-diagonal elements are zero
        for i in 1:5, j in 1:5
            if i != j
                @test mat[i, j] == 0.0f0
            end
        end

        # Test with gain parameter
        mat_gain = identity_init(4, 4; gain=2.5)
        @test mat_gain ≈ 2.5f0 * Matrix{Float32}(I, 4, 4)
        @test diag(mat_gain) == fill(2.5f0, 4)

        # Non-square matrices
        mat_rect1 = identity_init(3, 5)
        @test size(mat_rect1) == (3, 5)
        @test diag(mat_rect1) == ones(Float32, 3)
        @test mat_rect1[:, 4:5] == zeros(Float32, 3, 2)

        mat_rect2 = identity_init(5, 3)
        @test size(mat_rect2) == (5, 3)
        @test diag(mat_rect2) == ones(Float32, 3)
        @test mat_rect2[4:5, :] == zeros(Float32, 2, 3)
    end

    @testset "Non-identity sizes" begin
        @test identity_init(2, 3)[:, end] == zeros(Float32, 2)
        @test identity_init(3, 2; shift=1)[1, :] == zeros(Float32, 2)
        @test identity_init(1, 1, 3, 4)[:, :, :, end] == zeros(Float32, 1, 1, 3)
        @test identity_init(2, 1, 3, 3)[end, :, :, :] == zeros(Float32, 1, 3, 3)
        @test identity_init(1, 2, 3, 3)[:, end, :, :] == zeros(Float32, 1, 3, 3)
    end
end

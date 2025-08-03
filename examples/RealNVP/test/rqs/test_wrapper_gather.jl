using Test
using Random
using Statistics
using NNlib

# Include the wrapper implementation
include("../../src/rqs/rqs_wrapper_gather.jl")

@testset "RQS Wrapper Layer" begin
    @testset "Parameterization" begin
        # Test parameterization function
        K = 4
        D = 2
        B = 3

        # Generate random logits
        rng = Random.default_rng()
        Random.seed!(rng, 42)
        logits = randn(rng, Float32, 3K + 1, D, B)

        # Parameterize
        x_pos, y_pos, d = parameterize_rqs(logits)

        # Check shapes
        @test size(x_pos) == (K + 1, D, B)
        @test size(y_pos) == (K + 1, D, B)
        @test size(d) == (K + 1, D, B)

        # Check constraints
        @test all(x_pos[1, :, :] .== 0)  # Start at 0
        @test all(x_pos[end, :, :] .== 1)  # End at 1
        @test all(y_pos[1, :, :] .== 0)  # Start at 0
        @test all(y_pos[end, :, :] .== 1)  # End at 1
        @test all(d .> 0)  # Positive derivatives

        # Check monotonicity
        for dim in 1:D, batch in 1:B
            @test all(diff(x_pos[:, dim, batch]) .> 0)  # Strictly increasing
            @test all(diff(y_pos[:, dim, batch]) .> 0)  # Strictly increasing
        end
    end

    @testset "Forward Transformation" begin
        # Test forward transformation with parameterization
        K = 3
        D = 1
        B = 2

        # Generate random logits
        rng = Random.default_rng()
        Random.seed!(rng, 123)
        logits = randn(rng, Float32, 3K + 1, D, B)

        # Test inputs
        u = rand(rng, Float32, D, B)

        # Forward transformation
        v, log_det = rqs_forward(u, logits)

        # Check shapes
        @test size(v) == size(u)
        @test size(log_det) == size(u)

        # Check constraints
        @test all(0 .<= v .<= 1)
        @test all(isfinite, log_det)
    end

    @testset "Inverse Transformation" begin
        # Test inverse transformation with parameterization
        K = 3
        D = 1
        B = 2

        # Generate random logits
        rng = Random.default_rng()
        Random.seed!(rng, 456)
        logits = randn(rng, Float32, 3K + 1, D, B)

        # Test inputs
        v = rand(rng, Float32, D, B)

        # Inverse transformation
        u, log_det = rqs_inverse(v, logits)

        # Check shapes
        @test size(u) == size(v)
        @test size(log_det) == size(v)

        # Check constraints
        @test all(0 .<= u .<= 1)
        @test all(isfinite, log_det)
    end

    @testset "Round-trip Accuracy" begin
        # Test round-trip accuracy with parameterization
        K = 4
        D = 2
        B = 3

        # Generate random logits
        rng = Random.default_rng()
        Random.seed!(rng, 789)
        logits = randn(rng, Float32, 3K + 1, D, B)

        # Test inputs
        u_original = rand(rng, Float32, D, B)

        # Forward then inverse
        v, _ = rqs_forward(u_original, logits)
        u_recovered, _ = rqs_inverse(v, logits)

        # Check round-trip accuracy
        max_error = maximum(abs.(u_original - u_recovered))
        @test max_error < 1e-5
    end

    @testset "RQSBijector Interface" begin
        # Test the RQSBijector struct interface
        K = 3
        bj = RQSBijector(K)

        # Generate test data
        rng = Random.default_rng()
        Random.seed!(rng, 999)
        logits = randn(rng, Float32, 3K + 1, 1, 1)
        u = rand(rng, Float32, 1, 1)

        # Test forward
        v, log_det_forward = forward_and_log_det(bj, u, logits)
        @test size(v) == size(u)
        @test size(log_det_forward) == size(u)

        # Test inverse
        u_recovered, log_det_inverse = inverse_and_log_det(bj, v, logits)
        @test size(u_recovered) == size(v)
        @test size(log_det_inverse) == size(v)

        # Check round-trip
        @test abs(u[1] - u_recovered[1]) < 1e-5
    end

    @testset "Multiple Dimensions" begin
        # Test with multiple dimensions and batches
        K = 5
        D = 3
        B = 4

        # Generate random logits
        rng = Random.default_rng()
        Random.seed!(rng, 111)
        logits = randn(rng, Float32, 3K + 1, D, B)

        # Test inputs
        u = rand(rng, Float32, D, B)

        # Forward transformation
        v, log_det_forward = rqs_forward(u, logits)

        # Inverse transformation
        u_recovered, log_det_inverse = rqs_inverse(v, logits)

        # Check shapes
        @test size(v) == size(u)
        @test size(u_recovered) == size(u)
        @test size(log_det_forward) == size(u)
        @test size(log_det_inverse) == size(u)

        # Check constraints
        @test all(0 .<= v .<= 1)
        @test all(0 .<= u_recovered .<= 1)
        @test all(isfinite, log_det_forward)
        @test all(isfinite, log_det_inverse)

        # Check round-trip accuracy
        max_error = maximum(abs.(u - u_recovered))
        @test max_error < 1e-5
    end

    @testset "Edge Cases" begin
        # Test edge cases and numerical stability
        K = 2
        D = 1
        B = 1

        # Test with extreme logits
        logits = Float32[5.0, -5.0, 5.0, -5.0, 0.0, 0.0, 0.0]  # Less extreme values
        u = Float32[0.5]

        # Should not produce NaN or Inf
        v, log_det = rqs_forward(u, logits)

        @test !any(isnan, v)
        @test !any(isinf, v)
        @test !any(isnan, log_det)
        @test !any(isinf, log_det)

        # Test with boundary inputs
        u_boundary = Float32[0.0, 1.0]
        v_boundary, _ = rqs_forward(u_boundary, logits)

        @test abs(v_boundary[1]) < 1e-6  # Should map 0 to 0
        @test abs(v_boundary[2] - 1.0) < 1e-6  # Should map 1 to 1
    end

    @testset "Broadcasting" begin
        # Test broadcasting with different input shapes
        K = 3
        D = 1
        B = 1

        # Generate logits
        rng = Random.default_rng()
        Random.seed!(rng, 222)
        logits = randn(rng, Float32, 3K + 1, D, B)

        # Test scalar input
        u_scalar = Float32[0.5]
        v_scalar, _ = rqs_forward(u_scalar, logits)
        @test length(v_scalar) == 1

        # Test vector input
        u_vector = Float32[0.2, 0.5, 0.8]
        v_vector, _ = rqs_forward(u_vector, logits)
        @test length(v_vector) == 3

        # Test matrix input
        u_matrix = reshape(Float32[0.2, 0.5, 0.8, 0.3], 2, 2)
        v_matrix, _ = rqs_forward(u_matrix, logits)
        @test size(v_matrix) == (2, 2)
    end
end

println("✅ All RQS wrapper layer tests passed!")

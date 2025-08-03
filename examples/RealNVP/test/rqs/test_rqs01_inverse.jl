using Test
using Random
using Statistics
using NNlib

# Include the RQS implementations
include("../../src/rqs/rqs01_forward.jl")
include("../../src/rqs/rqs01_inverse.jl")

@testset "RQS Inverse Transformation" begin
    @testset "Round-trip Accuracy" begin
        # Test that forward then inverse gives original input
        x_pos = [0.0, 0.3, 0.7, 1.0]
        y_pos = [0.0, 0.2, 0.8, 1.0]
        d = [1.5, 2.0, 1.8, 1.5]

        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)

        u_test = [0.1, 0.5, 0.9]
        v_test, _ = rqs01_forward(u_test, x_pos_3d, y_pos_3d, d_3d)
        u_recovered, _ = rqs01_inverse(v_test, x_pos_3d, y_pos_3d, d_3d)

        # Check round-trip accuracy
        for i in 1:length(u_test)
            @test abs(u_test[i] - u_recovered[i]) < 1e-6
        end
    end

    @testset "Derivative Reciprocity" begin
        # Test that (∂v/∂u) * (∂u/∂v) ≈ 1
        x_pos = [0.0, 0.5, 1.0]
        y_pos = [0.0, 0.5, 1.0]
        d = [1.5, 2.0, 1.5]

        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)

        u_test = [0.3]
        _, log_det_forward = rqs01_forward(u_test, x_pos_3d, y_pos_3d, d_3d)
        _, log_det_inverse = rqs01_inverse(u_test, x_pos_3d, y_pos_3d, d_3d)

        # Check reciprocity
        forward_deriv = exp(log_det_forward[1])
        inverse_deriv = exp(log_det_inverse[1])
        reciprocity = forward_deriv * inverse_deriv

        @test abs(reciprocity - 1.0) < 1e-6
    end

    @testset "Boundary Conditions" begin
        # Test inverse at boundaries
        x_pos = [0.0, 0.5, 1.0]
        y_pos = [0.0, 0.5, 1.0]
        d = [1.0, 1.0, 1.0]

        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)

        u0, _ = rqs01_inverse([0.0], x_pos_3d, y_pos_3d, d_3d)
        u1, _ = rqs01_inverse([1.0], x_pos_3d, y_pos_3d, d_3d)

        @test abs(u0[1]) < 1e-6
        @test abs(u1[1] - 1.0) < 1e-6
    end

    @testset "Linear Case (aL = aR)" begin
        # When slopes are equal, inverse should be approximately linear
        x_pos = [0.0, 0.5, 1.0]
        y_pos = [0.0, 0.5, 1.0]
        d = [1.0, 1.0, 1.0]  # Equal derivatives

        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)

        v_test = [0.25, 0.75]
        u_test, _ = rqs01_inverse(v_test, x_pos_3d, y_pos_3d, d_3d)

        # Should be close to linear transformation
        @test abs(u_test[1] - 0.25) < 1e-3
        @test abs(u_test[2] - 0.75) < 1e-3
    end

    @testset "Monotonicity" begin
        # Test that inverse function is strictly increasing
        x_pos = [0.0, 0.3, 0.7, 1.0]
        y_pos = [0.0, 0.2, 0.8, 1.0]
        d = [1.5, 2.0, 1.8, 1.5]  # Positive derivatives

        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)

        v_vals = collect(0.0:0.1:1.0)
        u_vals, _ = rqs01_inverse(v_vals, x_pos_3d, y_pos_3d, d_3d)

        # Check monotonicity
        for i in 2:length(u_vals)
            @test u_vals[i] > u_vals[i-1]
        end
    end

    @testset "Newton-Raphson Convergence" begin
        # Test convergence for various inputs
        x_pos = [0.0, 0.5, 1.0]
        y_pos = [0.0, 0.5, 1.0]
        d = [1.5, 2.0, 1.5]

        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)

        # Test various input values
        v_test = [0.1, 0.3, 0.5, 0.7, 0.9]
        u_test, _ = rqs01_inverse(v_test, x_pos_3d, y_pos_3d, d_3d)

        # All should be finite and in [0,1]
        @test all(isfinite, u_test)
        @test all(0 .<= u_test .<= 1)
    end

    @testset "Broadcasting" begin
        # Test broadcasting over different shapes
        x_pos = [0.0, 0.5, 1.0]
        y_pos = [0.0, 0.5, 1.0]
        d = [1.0, 1.0, 1.0]

        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)

        # Test scalar input
        u_scalar, _ = rqs01_inverse([0.5], x_pos_3d, y_pos_3d, d_3d)

        # Test vector input
        u_vector, _ = rqs01_inverse([0.2, 0.5, 0.8], x_pos_3d, y_pos_3d, d_3d)

        # Test matrix input
        v_matrix = reshape([0.2, 0.5, 0.8, 0.3], 2, 2)
        u_matrix, _ = rqs01_inverse(v_matrix, x_pos_3d, y_pos_3d, d_3d)

        # All should work without errors
        @test length(u_scalar) == 1
        @test length(u_vector) == 3
        @test size(u_matrix) == (2, 2)
    end

    @testset "Numerical Stability" begin
        # Test edge cases that might cause numerical issues
        x_pos = [0.0, 1e-10, 1.0]  # Very small bin
        y_pos = [0.0, 1e-10, 1.0]
        d = [1.0, 1.0, 1.0]

        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)

        # Should not produce NaN or Inf
        u, log_det = rqs01_inverse([0.5], x_pos_3d, y_pos_3d, d_3d)

        @test !any(isnan, u)
        @test !any(isinf, u)
        @test !any(isnan, log_det)
        @test !any(isinf, log_det)
    end

    @testset "Multiple Dimensions" begin
        # Test with multiple dimensions and batches
        K = 4
        D = 2
        B = 3

        # Generate random parameters
        rng = Random.default_rng()
        Random.seed!(rng, 42)

        # Create valid spline parameters
        widths = softmax(randn(rng, Float32, K, D, B); dims = 1)
        heights = softmax(randn(rng, Float32, K, D, B); dims = 1)
        derivatives = softplus.(randn(rng, Float32, K + 1, D, B))

        x_pos = vcat(zeros(1, D, B), cumsum(widths; dims = 1))
        y_pos = vcat(zeros(1, D, B), cumsum(heights; dims = 1))
        d = derivatives

        # Test inputs
        v = rand(rng, Float32, D, B)

        u, log_det = rqs01_inverse(v, x_pos, y_pos, d)

        # Check output shapes
        @test size(u) == size(v)
        @test size(log_det) == size(v)

        # Check that outputs are in [0,1]
        @test all(0 .<= u .<= 1)

        # Check that log_det is finite
        @test all(isfinite, log_det)
    end

    @testset "Full Round-trip" begin
        # Comprehensive round-trip test
        K = 6
        D = 2
        B = 4

        # Generate random parameters
        rng = Random.default_rng()
        Random.seed!(rng, 123)

        # Create valid spline parameters
        widths = softmax(randn(rng, Float32, K, D, B); dims = 1)
        heights = softmax(randn(rng, Float32, K, D, B); dims = 1)
        derivatives = softplus.(randn(rng, Float32, K + 1, D, B))

        x_pos = vcat(zeros(1, D, B), cumsum(widths; dims = 1))
        y_pos = vcat(zeros(1, D, B), cumsum(heights; dims = 1))
        d = derivatives

        # Test inputs
        u_original = rand(rng, Float32, D, B)

        # Forward transformation
        v, log_det_forward = rqs01_forward(u_original, x_pos, y_pos, d)

        # Inverse transformation
        u_recovered, log_det_inverse = rqs01_inverse(v, x_pos, y_pos, d)

        # Check round-trip accuracy
        max_error = maximum(abs.(u_original - u_recovered))
        @test max_error < 1e-5

        # Check derivative reciprocity
        reciprocity_error = maximum(abs.(exp.(log_det_forward) .* exp.(log_det_inverse) .- 1.0))
        @test reciprocity_error < 1e-5
    end
end

println("✅ All RQS inverse transformation tests passed!")

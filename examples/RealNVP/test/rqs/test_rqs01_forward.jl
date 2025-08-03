using Test
using Random
using Statistics
using NNlib

# Include the RQS implementation
include("../../src/rqs/rqs01_forward.jl")

@testset "RQS Forward Transformation" begin
    @testset "Boundary Conditions" begin
        # Test v(0) = 0 and v(1) = 1
        x_pos = [0.0, 0.5, 1.0]
        y_pos = [0.0, 0.5, 1.0]
        d = [1.0, 1.0, 1.0]
        
        # Reshape for 3D input
        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)
        
        v0, _ = rqs01_forward([0.0], x_pos_3d, y_pos_3d, d_3d)
        v1, _ = rqs01_forward([1.0], x_pos_3d, y_pos_3d, d_3d)
        
        @test abs(v0[1]) < 1e-6
        @test abs(v1[1] - 1.0) < 1e-6
    end
    
    @testset "Linear Case (aL = aR)" begin
        # When slopes are equal, should be approximately linear
        x_pos = [0.0, 0.5, 1.0]
        y_pos = [0.0, 0.5, 1.0]
        d = [1.0, 1.0, 1.0]  # Equal derivatives
        
        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)
        
        u_test = [0.25, 0.75]
        v_test, _ = rqs01_forward(u_test, x_pos_3d, y_pos_3d, d_3d)
        
        # Should be close to linear transformation
        @test abs(v_test[1] - 0.25) < 1e-3
        @test abs(v_test[2] - 0.75) < 1e-3
    end
    
    @testset "Monotonicity" begin
        # Test that function is strictly increasing
        x_pos = [0.0, 0.3, 0.7, 1.0]
        y_pos = [0.0, 0.2, 0.8, 1.0]
        d = [1.5, 2.0, 1.8, 1.5]  # Positive derivatives
        
        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)
        
        u_vals = collect(0.0:0.1:1.0)
        v_vals, _ = rqs01_forward(u_vals, x_pos_3d, y_pos_3d, d_3d)
        
        # Check monotonicity
        for i in 2:length(v_vals)
            @test v_vals[i] > v_vals[i-1]
        end
    end
    
    @testset "Finite Differences" begin
        # Test that finite differences match analytical derivatives
        x_pos = [0.0, 0.5, 1.0]
        y_pos = [0.0, 0.5, 1.0]
        d = [1.5, 2.0, 1.5]
        
        x_pos_3d = reshape(x_pos, :, 1, 1)
        y_pos_3d = reshape(y_pos, :, 1, 1)
        d_3d = reshape(d, :, 1, 1)
        
        u0 = 0.3
        h = 1e-6
        
        # Finite difference approximation
        v_plus, log_det_plus = rqs01_forward([u0 + h], x_pos_3d, y_pos_3d, d_3d)
        v_minus, log_det_minus = rqs01_forward([u0 - h], x_pos_3d, y_pos_3d, d_3d)
        
        finite_diff = (v_plus[1] - v_minus[1]) / (2 * h)
        
        # Analytical derivative from log_det
        _, log_det = rqs01_forward([u0], x_pos_3d, y_pos_3d, d_3d)
        analytical_diff = exp(log_det[1])
        
        @test abs(finite_diff - analytical_diff) < 1e-3
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
        v_scalar, _ = rqs01_forward([0.5], x_pos_3d, y_pos_3d, d_3d)
        
        # Test vector input
        v_vector, _ = rqs01_forward([0.2, 0.5, 0.8], x_pos_3d, y_pos_3d, d_3d)
        
        # Test matrix input
        u_matrix = reshape([0.2, 0.5, 0.8, 0.3], 2, 2)
        v_matrix, _ = rqs01_forward(u_matrix, x_pos_3d, y_pos_3d, d_3d)
        
        # All should work without errors
        @test length(v_scalar) == 1
        @test length(v_vector) == 3
        @test size(v_matrix) == (2, 2)
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
        v, log_det = rqs01_forward([0.5], x_pos_3d, y_pos_3d, d_3d)
        
        @test !any(isnan, v)
        @test !any(isinf, v)
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
        widths = softmax(randn(rng, Float32, K, D, B); dims=1)
        heights = softmax(randn(rng, Float32, K, D, B); dims=1)
        derivatives = softplus.(randn(rng, Float32, K+1, D, B))
        
        x_pos = vcat(zeros(1, D, B), cumsum(widths; dims=1))
        y_pos = vcat(zeros(1, D, B), cumsum(heights; dims=1))
        d = derivatives
        
        # Test inputs
        u = rand(rng, Float32, D, B)
        
        v, log_det = rqs01_forward(u, x_pos, y_pos, d)
        
        # Check output shapes
        @test size(v) == size(u)
        @test size(log_det) == size(u)
        
        # Check that outputs are in [0,1]
        @test all(0 .<= v .<= 1)
        
        # Check that log_det is finite
        @test all(isfinite, log_det)
    end
end

println("✅ All RQS forward transformation tests passed!") 
include("bmm_testsetup.jl")

using Reactant, Enzyme, LuxLib, NNlib, Test
using LuxLib: batched_matmul
using NNlib: batched_mul

@testset "Generalized Batched MatMul" begin
    @testset "Last 2 dims are batch dims" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 3, 4, 5, 2)
        y = Reactant.TestUtils.construct_test_array(Float32, 5, 4, 5, 1)

        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        bmm(x, y) = batched_matmul(x, y; lhs_contracting_dim=2, rhs_contracting_dim=2)

        bmm_nnlib = batched_mul(x, repeat(permutedims(y, (2, 1, 3, 4)), 1, 1, 1, 2))
        bmm_luxlib = bmm(x, y)
        bmm_reactant = @jit bmm(x_ra, y_ra)

        hlo = @code_hlo bmm(x_ra, y_ra)
        @test !contains(repr(hlo), "transpose")

        @test bmm_nnlib ≈ bmm_luxlib atol = 1.0e-3 rtol = 1.0e-3
        @test bmm_luxlib ≈ bmm_reactant atol = 1.0e-3 rtol = 1.0e-3

        ∂x_fd, ∂y_fd = @jit Reactant.TestUtils.finite_difference_gradient(
            sum ∘ bmm, Float64.(x_ra), Float64.(y_ra)
        )
        ∂x_reactant, ∂y_reactant = @jit Enzyme.gradient(Reverse, sum ∘ bmm, x_ra, y_ra)

        @test ∂x_fd ≈ ∂x_reactant atol = 1.0e-3 rtol = 1.0e-3
        @test ∂y_fd ≈ ∂y_reactant atol = 1.0e-3 rtol = 1.0e-3
    end

    @testset "Middle dims are batch dims" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 3, 5, 2, 4)
        y = Reactant.TestUtils.construct_test_array(Float32, 4, 5, 1, 5)

        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        bmm(x, y) = batched_matmul(
            x,
            y;
            lhs_contracting_dim=4,
            rhs_contracting_dim=1,
            lhs_batching_dims=(2, 3),
            rhs_batching_dims=(4, 3),
        )

        bmm_nnlib = batched_mul(
            permutedims(x, (1, 4, 2, 3)), repeat(permutedims(y, (1, 2, 4, 3)), 1, 1, 1, 2)
        )
        bmm_luxlib = bmm(x, y)
        bmm_reactant = @jit bmm(x_ra, y_ra)

        @test bmm_nnlib ≈ bmm_luxlib atol = 1.0e-3 rtol = 1.0e-3
        @test bmm_luxlib ≈ bmm_reactant atol = 1.0e-3 rtol = 1.0e-3

        ∂x_fd, ∂y_fd = @jit Reactant.TestUtils.finite_difference_gradient(
            sum ∘ bmm, Float64.(x_ra), Float64.(y_ra)
        )
        ∂x_reactant, ∂y_reactant = @jit Enzyme.gradient(Reverse, sum ∘ bmm, x_ra, y_ra)

        @test ∂x_fd ≈ ∂x_reactant atol = 1.0e-3 rtol = 1.0e-3
        @test ∂y_fd ≈ ∂y_reactant atol = 1.0e-3 rtol = 1.0e-3
    end
end

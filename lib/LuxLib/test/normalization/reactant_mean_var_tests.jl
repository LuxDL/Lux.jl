include("../shared_testsetup.jl")

using LuxLib, Reactant, Enzyme, Statistics, StableRNGs, Test, Zygote

# The Reactant method of `LuxLib.Impl.mean_var` detaches the mean inside the variance, which removes
# the identically zero ∂σ²/∂μ term from the reverse pass. These tests pin what must NOT change: the
# statistics themselves, and the gradients of the normalizations built on them. The gradient
# reference is Zygote on host arrays, which uses LuxLib's hand-written ChainRules backward and never
# reaches the Reactant method.

@testset "Reactant mean_var" begin
    rng = StableRNG(0)

    @testset "statistics match Statistics (dims = $dims, corrected = $corrected)" for dims in (
            1, (1, 2), (1, 2, 3),
        ),
        corrected in (false, true)

        x = randn(rng, Float32, 5, 6, 8, 3) .+ 2.0f0
        f = x -> LuxLib.Impl.mean_var(x; dims, corrected)
        μ, σ² = f(x)
        μ_ra, σ²_ra = @jit f(Reactant.to_rarray(x))
        @test μ_ra ≈ μ atol = 1.0f-5 rtol = 1.0f-5
        @test σ²_ra ≈ σ² atol = 1.0f-5 rtol = 1.0f-5
        @test μ ≈ mean(x; dims) && σ² ≈ var(x; dims, corrected)
    end

    # Non-uniform weights on the output, so every gradient path through the statistics matters: a
    # uniform weight would make the mean-subtraction term vanish and hide a wrong gradient.
    @testset "groupnorm gradient matches LuxLib's ChainRules backward" begin
        x = randn(rng, Float64, 4, 5, 8, 2) .+ 1.0
        γ, β = randn(rng, Float64, 8), randn(rng, Float64, 8)
        w = randn(rng, Float64, 4, 5, 8, 2)
        loss(x, γ, β, w) = sum(groupnorm(x, γ, β, 4, identity, 1.0e-5) .* w)
        ∂_ref = Zygote.gradient(loss, x, γ, β, w)
        ∂_ra = @jit Enzyme.gradient(Reverse, Const(loss), Reactant.to_rarray.((x, γ, β, w))...)
        for (ref, ra) in zip(∂_ref, ∂_ra)
            @test Array(ra) ≈ ref atol = 1.0e-10 rtol = 1.0e-10
        end
    end

    @testset "layernorm gradient matches LuxLib's ChainRules backward" begin
        x = randn(rng, Float64, 6, 5, 3) .+ 1.0
        γ, β = randn(rng, Float64, 6, 1, 1), randn(rng, Float64, 6, 1, 1)
        w = randn(rng, Float64, 6, 5, 3)
        loss(x, γ, β, w) = sum(layernorm(x, γ, β, identity, 1, 1.0e-5) .* w)
        ∂_ref = Zygote.gradient(loss, x, γ, β, w)
        ∂_ra = @jit Enzyme.gradient(Reverse, Const(loss), Reactant.to_rarray.((x, γ, β, w))...)
        for (ref, ra) in zip(∂_ref, ∂_ra)
            @test Array(ra) ≈ ref atol = 1.0e-10 rtol = 1.0e-10
        end
    end
end

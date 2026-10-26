include("../shared_testsetup.jl")

using LuxLib, Reactant, Enzyme, ForwardDiff, Statistics, StableRNGs, Test, Zygote

# The Reactant method of `LuxLib.Impl.mean_var` computes the variance about a detached copy of the
# mean plus an exact correction, which removes the identically zero ∂σ²/∂μ term from the reverse
# pass. These tests pin what must NOT change: the statistics themselves, and the first and second
# derivatives of the normalizations built on them. The gradient reference is Zygote on host arrays,
# which uses LuxLib's hand-written ChainRules backward and never reaches the Reactant method; the
# second-order reference is nested ForwardDiff on a plain Statistics implementation.

function groupnorm_reference(x, γ, β, groups, ϵ)
    W, H, C, N = size(x)
    xr = reshape(x, W, H, C ÷ groups, groups, N)
    μ = mean(xr; dims=(1, 2, 3))
    σ² = var(xr; dims=(1, 2, 3), corrected=false, mean=μ)
    x̂ = reshape((xr .- μ) ./ sqrt.(σ² .+ ϵ), W, H, C, N)
    return x̂ .* reshape(γ, 1, 1, C, 1) .+ reshape(β, 1, 1, C, 1)
end

function layernorm_reference(x, γ, β, dims, ϵ)
    μ = mean(x; dims)
    σ² = var(x; dims, corrected=false, mean=μ)
    return γ .* (x .- μ) ./ sqrt.(σ² .+ ϵ) .+ β
end

function ∇ₓ(loss, x, γ, β, w)
    return Enzyme.gradient(Reverse, Const(loss), x, Const(γ), Const(β), Const(w))[1]
end

# Forward over reverse
function hvp_forward_over_reverse(loss, x, v, γ, β, w)
    return Enzyme.autodiff(Forward, Const(x -> ∇ₓ(loss, x, γ, β, w)), Duplicated(x, v))[1]
end

# Reverse over reverse, the shape of a gradient penalty: ∇ₓ⟨∇ₓ loss, v⟩
∇ₓ_dot(loss, x, v, γ, β, w) = sum(∇ₓ(loss, x, γ, β, w) .* v)
function hvp_reverse_over_reverse(loss, x, v, γ, β, w)
    return Enzyme.gradient(
        Reverse, Const(∇ₓ_dot), Const(loss), x, Const(v), Const(γ), Const(β), Const(w)
    )[2]
end

@testset "Reactant mean_var" begin
    rng = StableRNG(0)

    all_dims = (1, (1, 2), (1, 2, 3), 1:2, :)
    @testset "statistics (dims = $dims, corrected = $corrected)" for dims in all_dims,
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
        ∂_ra = @jit Enzyme.gradient(
            Reverse, Const(loss), Reactant.to_rarray.((x, γ, β, w))...
        )
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
        ∂_ra = @jit Enzyme.gradient(
            Reverse, Const(loss), Reactant.to_rarray.((x, γ, β, w))...
        )
        for (ref, ra) in zip(∂_ref, ∂_ra)
            @test Array(ra) ≈ ref atol = 1.0e-10 rtol = 1.0e-10
        end
    end

    # Second order, forward over reverse and reverse over reverse. A variance that merely detaches
    # the mean loses the -1/M term of its Hessian wherever the detach holds at every order; the
    # correction keeps it exact. Small reductions make that term large.
    @testset "groupnorm and layernorm HVPs match nested ForwardDiff" begin
        x_gn = randn(rng, Float64, 4, 5, 8, 2) .+ 1.0
        γ_gn, β_gn = randn(rng, Float64, 8), randn(rng, Float64, 8)
        w_gn = randn(rng, Float64, 4, 5, 8, 2)
        x_ln = randn(rng, Float64, 6, 5, 3) .+ 1.0
        γ_ln, β_ln = randn(rng, Float64, 6, 1, 1), randn(rng, Float64, 6, 1, 1)
        w_ln = randn(rng, Float64, 6, 5, 3)

        cases = (
            (
                (x, γ, β, w) -> sum(groupnorm(x, γ, β, 4, identity, 1.0e-5) .* w),
                (x, γ, β, w) -> sum(groupnorm_reference(x, γ, β, 4, 1.0e-5) .* w),
                x_gn,
                γ_gn,
                β_gn,
                w_gn,
            ),
            (
                (x, γ, β, w) -> sum(layernorm(x, γ, β, identity, 1, 1.0e-5) .* w),
                (x, γ, β, w) -> sum(layernorm_reference(x, γ, β, 1, 1.0e-5) .* w),
                x_ln,
                γ_ln,
                β_ln,
                w_ln,
            ),
        )
        for (loss, loss_ref, x, γ, β, w) in cases
            v = randn(rng, Float64, size(x)...)
            hvp_ref = ForwardDiff.derivative(
                t -> ForwardDiff.gradient(z -> loss_ref(z, γ, β, w), x .+ t .* v), 0.0
            )
            args_ra = Reactant.to_rarray.((x, v, γ, β, w))
            @test Array(@jit hvp_forward_over_reverse(loss, args_ra...)) ≈ hvp_ref atol =
                1.0e-10 rtol = 1.0e-10
            @test Array(@jit hvp_reverse_over_reverse(loss, args_ra...)) ≈ hvp_ref atol =
                1.0e-10 rtol = 1.0e-10
        end
    end
end

# Reverse-mode AD of `var(x; mean = μ)` differentiates through `μ` as well as `x`, and the term it
# gets through `μ` is ∂σ²/∂μ = -2/(M - c) Σ(x - μ), which is identically zero because `μ` is the mean
# of `x`. Enzyme cannot see that, so every normalization's backward pass spends a full-size
# multiply-and-reduce computing it. We compute the variance about a detached copy μ₀ of the mean
# instead, and subtract the König-Huygens correction, which is exact for any constant shift:
#
#     σ² = Σ(x - μ₀)² / (M - c) - M / (M - c) * (μ - μ₀)²
#
# The correction is zero in value (μ₀ = μ) and only involves the reduced-size `μ`, so the full-size
# reduction is gone. Unlike detaching `μ` alone, this is still an exact formula for the variance as a
# function of `x`, so derivatives of every order (nested AD, gradient penalties) stay correct
# regardless of how `ignore_derivatives` behaves under nesting.
function Impl.mean_var(x::AnyTracedRArray; dims=:, corrected::Bool=true)
    μ = mean(x; dims)
    μ₀ = EnzymeCore.ignore_derivatives(μ)
    M = length(x) ÷ length(μ)  # elements per reduction; sizes are static at trace time
    k = Reactant.unwrapped_eltype(x)(M / (M - corrected))
    return μ, var(x; dims, corrected, mean=μ₀) .- k .* abs2.(μ .- μ₀)
end

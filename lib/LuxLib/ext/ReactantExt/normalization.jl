# Reverse-mode AD of `var(x; mean = μ)` differentiates through `μ` as well as `x`, and the term it
# gets through `μ` is ∂σ²/∂μ = -(2/M) Σ(x - μ), which is identically zero because `μ` is the mean of
# `x`. Enzyme cannot see that, so every normalization's backward pass spends a full-size
# multiply-and-reduce computing it. Detaching `μ` inside the variance removes that work and nothing
# else: the forward pass is unchanged, and the gradient through `μ` (which the normalization uses
# directly) and through `σ²` to `x` is exactly the same.
function Impl.mean_var(x::AnyTracedRArray; dims=:, corrected::Bool=true)
    μ = mean(x; dims)
    return μ, var(x; dims, corrected, mean=Reactant.ignore_derivatives(μ))
end

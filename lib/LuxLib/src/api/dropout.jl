"""
    dropout(rng::AbstractRNG, x, p, training::Union{Val, StaticBool}, invp, dims)
    dropout(rng::AbstractRNG, x, mask, p, training::Union{Val, StaticBool},
        update_mask::Union{Val, StaticBool}, invp, dims)

Dropout: Simple Way to prevent Neural Networks for Overfitting. For details see [1].

## Arguments

  - `rng`: Random number generator
  - `x`: Input Array
  - `mask`: Dropout Mask. If not used then it is constructed automatically
  - `p`: Probability of an element to be dropped out
  - `Val(training)`: If `true` then dropout is applied on `x` with probability `p` along
    `dims`. Else, `x` is returned
  - `Val(update_mask)`: If `true` then the mask is generated and used. Else, the `mask`
    provided is directly used
  - `invp`: Inverse multiplied to the mask. Calculated as `invp = 1 / (1 - p)`.

## Returns

  - Output Array after applying dropout
  - Dropout Mask (if `training == false`, the returned value is meaningless)
  - Updated state for the random number generator

## References

[1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from
overfitting." The journal of machine learning research 15.1 (2014): 1929-1958.
"""
function dropout(rng::AbstractRNG, x::AbstractArray, p::T,
        training::Union{Val, StaticBool}, invp::T, dims) where {T}
    return Impl.dropout(rng, x, p, static(training), invp, dims)
end

function dropout(rng::AbstractRNG, x::AbstractArray, mask::AbstractArray,
        p::T, update_mask::Union{Val, StaticBool},
        training::Union{Val, StaticBool}, invp::T, dims) where {T}
    return Impl.dropout(rng, x, mask, p, static(update_mask), static(training), invp, dims)
end

"""
    alpha_dropout(rng::AbstractRNG, x, p, training::Union{Val, StaticBool})
    alpha_dropout(rng::AbstractRNG, x, p, training::Union{Val, StaticBool}, α, A, B)

Alpha Dropout: Dropout ensuring that the mean and variance of the output remains same as the
input. For details see [1]. Use the second call signature to avoid recomputing the constants
for a fixed dropout probability.

## Arguments

  - `rng`: Random number generator
  - `x`: Input Array
  - `p`: Probability of an element to be dropped out
  - `Val(training)`: If `true` then dropout is applied on `x` with probability `p`. Else,
    `x` is returned
  - `α`: `-1.7580993408473766`. Computed at limit x tends to infinity, `selu(x) = -λβ = α`
  - `A`: Scaling factor for the mean
  - `B`: Scaling factor for the variance

## Returns

  - Output Array after applying alpha dropout
  - Updated state for the random number generator

## References

[1] Klambauer, Günter, et al. "Self-normalizing neural networks." Advances in neural
information processing systems 30 (2017).
"""
function alpha_dropout(
        rng::AbstractRNG, x::AbstractArray, p, training::Union{Val, StaticBool})
    return Impl.alpha_dropout(rng, x, p, static(training))
end

function alpha_dropout(
        rng::AbstractRNG, x::AbstractArray, p, training::Union{Val, StaticBool}, α, A, B)
    return Impl.alpha_dropout(rng, x, p, static(training), α, A, B)
end

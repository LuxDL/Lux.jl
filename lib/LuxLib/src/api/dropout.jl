"""
    dropout(rng::AbstractRNG, x, p, training, invp, dims)
    dropout(rng::AbstractRNG, x, mask, p, training, update_mask::Union{Val, StaticBool},
        invp, dims)

Dropout: Simple Way to prevent Neural Networks for Overfitting. For details see [1].

## Arguments

  - `rng`: Random number generator
  - `x`: Input Array
  - `mask`: Dropout Mask. If not used then it is constructed automatically
  - `p`: Probability of an element to be dropped out
  - `training`: Set to `Val(true)` or `True()` if running in training mode. Can be set to
    `nothing` to automatically determine if the function is being called within an autodiff
    context
  - `update_mask`: If `Val(true)` or `True()` then the mask is generated and used. Else, the
    `mask` provided is directly used
  - `invp`: Inverse multiplied to the mask. Calculated as `invp = 1 / (1 - p)`.

## Returns

  - Output Array after applying dropout
  - Dropout Mask (if `training == false`, the returned value is meaningless)
  - Updated state for the random number generator

## References

[1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from
overfitting." The journal of machine learning research 15.1 (2014): 1929-1958.
"""
function dropout(rng::AbstractRNG, x::AbstractArray, p::T, training::TrainingType, invp::T,
        dims) where {T}
    return dropout_impl(rng, x, p, static_training_mode(training, x), invp, dims)
end

function dropout(rng::AbstractRNG, x::AbstractArray, mask::AbstractArray,
        p::T, training::TrainingType, update_mask::TrainingType, invp::T, dims) where {T}
    return dropout_impl(rng, x, mask, p, static_training_mode(training, x),
        static(update_mask), invp, dims)
end

"""
    alpha_dropout(rng::AbstractRNG, x, p, training)
    alpha_dropout(rng::AbstractRNG, x, p, training, α, A, B)

Alpha Dropout: Dropout ensuring that the mean and variance of the output remains same as the
input. For details see [1]. Use the second call signature to avoid recomputing the constants
for a fixed dropout probability.

## Arguments

  - `rng`: Random number generator
  - `x`: Input Array
  - `p`: Probability of an element to be dropped out
  - `training`: Set to `Val(true)` or `True()` if running in training mode. Can be set to
    `nothing` to automatically determine if the function is being called within an autodiff
    context`
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
function alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, training::TrainingType)
    return alpha_dropout_impl(rng, x, p, static_training_mode(training, x))
end

function alpha_dropout(
        rng::AbstractRNG, x::AbstractArray, p, training::TrainingType, α, A, B)
    return alpha_dropout_impl(rng, x, p, static_training_mode(training, x), α, A, B)
end

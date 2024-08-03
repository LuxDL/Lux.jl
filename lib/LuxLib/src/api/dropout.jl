@doc doc"""
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
function dropout(
        rng::AbstractRNG, x::AbstractArray, p::T, training, invp::T, dims) where {T}
    return dropout(rng, x, p, static(training), invp, dims)
end

function dropout(rng::AbstractRNG, x::AbstractArray, p::T, ::True, invp::T, dims) where {T}
    mask, rng_new = _generate_dropout_mask(rng, x, p, invp; dims)
    return __dropout_dot_mul(x, mask), mask, rng_new
end

function dropout(rng::AbstractRNG, x::AbstractArray, ::T, ::False, ::T, dims) where {T}
    return (x, x, rng)
end

function dropout(rng::AbstractRNG, x::AbstractArray, mask::AbstractArray,
        p::T, update_mask, training, invp::T, dims) where {T}
    return dropout(rng, x, mask, p, static(update_mask), static(training), invp, dims)
end

function dropout(rng::AbstractRNG, x::AbstractArray, ::AbstractArray,
        p::T, training::StaticBool, ::True, invp::T, dims) where {T}
    return dropout(rng, x, p, training, invp, dims)
end

function dropout(rng::AbstractRNG, x::AbstractArray{T1, N}, mask::AbstractArray{T2, N},
        p::T, ::True, ::False, invp::T, dims) where {T, T1, T2, N}
    if _dropout_shape(x, dims) != size(mask)
        __depwarn("`update_mask` is `Val(false)` but `mask` is not of the same size as \
                   `LuxLib._dropout_shape(x, dims)`. This has been deprecated and will be \
                   removed in the next release. Set `update_mask` to `Val(true)` to \
                   avoid this.",
            :dropout)
        mask, rng_new = _generate_dropout_mask(rng, x, p, invp; dims)
        return __dropout_dot_mul(x, mask), mask, rng_new
    end
    return __dropout_dot_mul(x, mask), mask, rng
end

function dropout(rng::AbstractRNG, x::AbstractArray{T1, N}, mask::AbstractArray{T2, N},
        ::T, ::False, ::False, invp::T, dims) where {T, T1, T2, N}
    return (x, mask, rng)
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
function alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, training)
    return alpha_dropout(rng, x, p, static(training))
end

function alpha_dropout(rng::AbstractRNG, x::AbstractArray{T}, p, training::True) where {T}
    α = T(-1.7580993408473766)
    A = T(inv(sqrt((1 - p) * (1 + p * α^2))))
    B = T(-A * α * p)
    return alpha_dropout(rng, x, p, training, α, A, B)
end

function alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, training::False)
    return alpha_dropout(rng, x, p, training, 0, 0, 0)
end

function alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, training, α, A, B)
    return alpha_dropout(rng, x, p, static(training), α, A, B)
end

function alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, ::True, α, A, B)
    noise, rng = _alpha_dropout_noise(rng, x)
    return _alpha_dropout_kernel(noise, p, x, α, A, B), rng
end

function alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, ::False, α, A, B)
    return (x, rng)
end

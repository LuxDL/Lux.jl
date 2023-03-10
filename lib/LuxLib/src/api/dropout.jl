@doc doc"""
    dropout(rng::AbstractRNG, x, p, ::Val{training}; dims, invp=inv(p))
    dropout(rng::AbstractRNG, x, mask, p, ::Val{training}, ::Val{update_mask}; dims,
            invp=inv(p))

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

## Keyword Arguments

  - `dims`: Dimensions along which dropout is applied
  - `invp`: Inverse of the probability (``\frac{1}{p}``)

## Returns

  - Output Array after applying dropout
  - Dropout Mask (if `training == false`, the returned value is meaningless)
  - Updated state for the random number generator

## References

[1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from
    overfitting." The journal of machine learning research 15.1 (2014): 1929-1958.
"""
function dropout(rng::AbstractRNG, x::AbstractArray, p::T, ::Val{true}; dims,
                 invp::T=inv(p)) where {T}
    rng = _replicate(rng)
    mask = _generate_dropout_mask(rng, x, p, invp; dims)
    return (x .* ignore_derivatives(mask), mask, rng)
end

function dropout(rng::AbstractRNG, x::AbstractArray, p::T, ::Val{false}; dims,
                 invp::T=inv(p)) where {T}
    return (x, x, rng)
end

function dropout(rng::AbstractRNG, x::AbstractArray, mask::AbstractArray, p::T, t::Val,
                 ::Val{true}; dims, invp::T=inv(p)) where {T}
    return dropout(rng, x, p, t; dims, invp)
end

function dropout(rng::AbstractRNG, x::AbstractArray{T1, N}, mask::AbstractArray{T2, N},
                 p::T, ::Val{true}, ::Val{false}; dims, invp::T=inv(p)) where {T, T1, T2, N}
    if size(x) != size(mask)
        return dropout(rng, x, p, Val(true); dims, invp)
    end
    return x .* ignore_derivatives(mask), mask, rng
end

function dropout(rng::AbstractRNG, x::AbstractArray{T1, N}, mask::AbstractArray{T2, N},
                 p::T, ::Val{false}, ::Val{false}; dims,
                 invp::T=inv(p)) where {T, T1, T2, N}
    return (x, mask, rng)
end

@doc doc"""
    alpha_dropout(rng::AbstractRNG, x, p, ::Val{training})
    alpha_dropout(rng::AbstractRNG, x, p, ::Val{training}, α, A, B)

Alpha Dropout: Dropout ensuring that the mean and variance of the output remains same as the
input. For details see [1]. Use the second call signature to avoid recomputing the constants
for a fixed dropout probability.

## Arguments

  - `rng`: Random number generator
  - `x`: Input Array
  - `p`: Probability of an element to be dropped out
  - `Val(training)`: If `true` then dropout is applied on `x` with probability `p`. Else,
    `x` is returned
  - `α`: -1.7580993408473766. Computed at limit x tends to infinity, `selu(x) = -λβ = α`
  - `A`: Scaling factor for the mean
  - `B`: Scaling factor for the variance

## Returns

  - Output Array after applying alpha dropout
  - Updated state for the random number generator

## References

[1] Klambauer, Günter, et al. "Self-normalizing neural networks." Advances in neural
    information processing systems 30 (2017).
"""
function alpha_dropout(rng::AbstractRNG, x::AbstractArray{T}, p, t::Val{true}) where {T}
    α = T(-1.7580993408473766)
    A = T(inv(sqrt((1 - p) * (1 + p * α^2))))
    B = T(-A * α * p)

    return alpha_dropout(rng, x, p, t, α, A, B)
end

function alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, t::Val{false})
    return alpha_dropout(rng, x, p, t, 0, 0, 0)
end

function alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, ::Val{true}, α, A, B)
    rng = _replicate(rng)
    noise = rand!(rng, similar(x, _dropout_fptype(x)))
    return (A .* ifelse.(noise .> p, x, α) .+ B), rng
end

alpha_dropout(rng::AbstractRNG, x::AbstractArray, p, ::Val{false}, α, A, B) = (x, rng)

# Mask Generation
@inline _dropout_shape(s, ::Colon) = size(s)
@inline function _dropout_shape(s, dims)
    return tuple((i in dims ? si : 1 for (i, si) in enumerate(size(s)))...)
end

@inline _dropout_kernel(y, p, invp) = y > p ? invp : oftype(y, 0)

@inline _dropout_fptype(x) = float(real(eltype(x)))

@inline function _generate_dropout_mask(rng::AbstractRNG, x, p, invp; dims)
    realfptype = _dropout_fptype(x)
    y = rand!(rng, similar(x, realfptype, _dropout_shape(x, dims)))
    y .= _dropout_kernel.(y, p, invp)
    return y
end

CRC.@non_differentiable _generate_dropout_mask(::Any...)
CRC.@non_differentiable _dropout_shape(::Any...)

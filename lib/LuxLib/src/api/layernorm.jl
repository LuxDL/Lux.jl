@doc doc"""
    layernorm(x, scale, bias, σ = identity; dims, epsilon)

Layer Normalization. For details see [1].

Given an input array ``x``, this layer computes

```math
y = \frac{x - \mathbb{E}[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
```

and applies the activation function `σ` elementwise to `y`.

## Arguments

  - `x`: Input to be Normalized
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)
  - `σ`: Activation function (default: `identity`)

## Keyword Arguments

  - `dims`: Dimensions along which the mean and std of `x` is computed
  - `epsilon`: Value added to the denominator for numerical stability

## Returns

Normalized Array of same size as `x`.

## References

[1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv
    preprint arXiv:1607.06450 (2016).
"""
function layernorm(
        x::AbstractArray{<:Number, N}, scale::Union{Nothing, AbstractArray{<:Number, N}},
        bias::Union{Nothing, AbstractArray{<:Number, N}},
        σ::F=identity; dims, epsilon) where {N, F}
    _mean = mean(x; dims)
    _var = var(x; dims, mean=_mean, corrected=false)
    return _affine_normalize(σ, x, _mean, _var, scale, bias, epsilon)
end

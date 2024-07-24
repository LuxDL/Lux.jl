@doc doc"""
    layernorm(x, scale, bias, σ = identity, dims=Colon(),
        epsilon = eps(eltype(x)) ^ (5 / 7))

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
  - `dims`: Dimensions along which the mean and std of `x` is computed (default: `Colon()`)
  - `epsilon`: Value added to the denominator for numerical stability
    (default: `eps(eltype(x)) ^ (5 / 7)`)

## Returns

Normalized Array of same size as `x`.

## References

[1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv
    preprint arXiv:1607.06450 (2016).
"""
function layernorm(
        x::AbstractArray{<:Number, N}, scale::Optional{<:AbstractArray{<:Number, N}},
        bias::Optional{<:AbstractArray{<:Number, N}}, σ::F=identity,
        dims=Colon(), epsilon::Real=__default_epsilon(x)) where {N, F}
    μ, σ² = fast_mean_var(x; dims, corrected=false)
    return _affine_normalize(
        sleefpirates_activation(σ, x, scale, bias, μ, σ²), x, μ, σ², scale, bias, epsilon)
end

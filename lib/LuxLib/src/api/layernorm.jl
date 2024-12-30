@doc doc"""
    layernorm(x::AbstractArray{xT, N}, scale, bias, σ = identity, dims=1:(N - 1),
        epsilon = eps(eltype(x)) ^ (5 / 7)) where {xT, N}

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
  - `dims`: Dimensions along which the mean and std of `x` is computed. If `nothing` is
    passed, the dims are inferred based on the dimensions of scale and bias. For example,
    if `x` is `N` dimensional and `scale` and `bias` are `M` dimensional, then the dims
    will be `1:(N - M)`.
  - `epsilon`: Value added to the denominator for numerical stability
    (default: `eps(eltype(x)) ^ (5 / 7)`)

## Returns

Normalized Array of same size as `x`.

## References

[1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv
    preprint arXiv:1607.06450 (2016).
"""
function layernorm(x::AbstractArray{xT, N}, scale::Optional{<:AbstractArray},
        bias::Optional{<:AbstractArray}, σ::F=identity, dims=1:(N - 1),
        epsilon=default_epsilon(x)) where {F, xT, N}
    return layernorm_impl(
        x, scale, bias, select_fastest_activation(σ, x, scale, bias), dims, epsilon)
end

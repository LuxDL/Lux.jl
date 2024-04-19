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
        x::AbstractArray{T1, N}, scale::AbstractArray{T2, N}, bias::AbstractArray{T3, N},
        σ::F=identity; dims, epsilon) where {N, T1, T2, T3, F}
    _mean = mean(x; dims)
    _std = std(x; dims, mean=_mean, corrected=false)
    _scale = @. scale / (_std + epsilon)
    _bias = @. bias - _mean * _scale
    σ === identity && return @. _scale * x + _bias
    return @. σ(_scale * x + _bias)
end

function layernorm(
        x::AbstractArray, ::Nothing, ::Nothing, σ::F=identity; dims, epsilon) where {F}
    _mean = mean(x; dims)
    _std = std(x; dims, mean=_mean, corrected=false)
    σ === identity && return @. (x .- _mean) / (_std + epsilon)
    return @. σ((x .- _mean) / (_std + epsilon))
end

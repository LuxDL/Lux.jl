@doc doc"""
    layernorm(x, scale, bias; dims, epsilon)

Layer Normalization. For details see [1].

Given an input array ``x``, this layer computes

```math
y = \frac{x - \mathbb{E}[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
```

## Arguments

  - `x`: Input to be Normalized
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)

## Keyword Arguments

  - `dims`: Dimensions along which the mean and std of `x` is computed
  - `epsilon`: Value added to the denominator for numerical stability

## Returns

Normalized Array of same size as `x`.

## References

[1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv
    preprint arXiv:1607.06450 (2016).
"""
function layernorm(x::AbstractArray{<:Real, N}, scale::AbstractArray{<:Real, N},
                   bias::AbstractArray{<:Real, N}; dims, epsilon) where {N}
    _mean = mean(x; dims)
    _rstd = 1 ./ (std(x; dims, mean=_mean, corrected=false) .+ epsilon)

    return scale .* (x .- _mean) .* _rstd .+ bias
end

function layernorm(x::AbstractArray, ::Nothing, ::Nothing; dims, epsilon)
    _mean = mean(x; dims)
    _rstd = 1 ./ (std(x; dims, mean=_mean, corrected=false) .+ epsilon)

    return (x .- _mean) .* _rstd
end

__BATCHNORM_ARRAY_TYPE = Union{AbstractVector{<:Real}, Nothing}

@doc doc"""
    batchnorm(x, scale, bias, running_mean, running_var; momentum, epsilon, training)

Batch Normalization. For details see [1].

Batch Normalization computes the mean and variance for each
``D_1 \times ... \times D_{N - 2} \times 1 \times D_N`` input slice and normalises the input
accordingly.

## Arguments

  - `x`: Input to be Normalized
  - `scale`: Scale factor (``\gamma``) (can be `nothing`)
  - `bias`: Bias factor (``\beta``) (can be `nothing`)
  - `running_mean`: Running mean (can be `nothing`)
  - `running_var`: Running variance (can be `nothing`)

## Keyword Arguments

  - `momentum`: Momentum for updating running mean and variance
  - `epsilon`: Value added to the denominator for numerical stability
  - `training`: Set to `Val(true)` if running in training mode

## Returns

Normalized Array of same size as `x`. And a Named Tuple containing the updated running
mean and variance.

## Performance Considerations

If the input array is `2D`, `4D`, or `5D` `CuArray` with element types `Float16`, `Float32`
and `Float64`, then the CUDNN code path will be used. In all other cases, a broadcasting
fallback is used which is not highly optimized.

## References

[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network
    training by reducing internal covariate shift." International conference on machine
    learning. PMLR, 2015.
"""
function batchnorm(x::AbstractArray{<:Real, N}, scale::__BATCHNORM_ARRAY_TYPE,
                   bias::__BATCHNORM_ARRAY_TYPE, running_mean::__BATCHNORM_ARRAY_TYPE,
                   running_var::__BATCHNORM_ARRAY_TYPE; momentum::Real, training::Val,
                   epsilon::Real) where {N}
    x_, xm, xv = _normalization(x, running_mean, running_var, scale, bias,
                                _get_batchnorm_reduce_dims(x), training, momentum, epsilon)

    return x_, (; running_mean=xm, running_var=xv)
end

@generated function _get_batchnorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return :($(Val(Tuple(collect([1:(N - 2); N])))))
end

# CUDNN dispatches
function _batchnorm_cudnn! end
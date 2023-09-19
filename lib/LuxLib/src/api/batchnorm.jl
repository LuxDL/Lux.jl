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
function batchnorm(x::AA{<:Real, N}, scale::NOrAVR, bias::NOrAVR, running_mean::NOrAVR,
    running_var::NOrAVR; momentum::Real, training::Val, epsilon::Real) where {N}
    x_, xm, xv = _normalization(x, running_mean, running_var, scale, bias,
        _get_batchnorm_reduce_dims(x), training, momentum, epsilon)
    stats = (; running_mean=_drop_forwarddiff_partials(xm),
        running_var=_drop_forwarddiff_partials(xv))
    return (x_, stats)
end

@generated function _get_batchnorm_reduce_dims(::AA{T, N}) where {T, N}
    return :($(Val(Tuple(collect([1:(N - 2); N])))))
end

function _get_batchnorm_statistics(x, running_mean, running_var,
    ::Val{training}) where {training}
    if training
        # NNlib silently updates running_mean and running_var. Copying them!
        rm = _copy_autodiff_barrier(running_mean)
        rv = _copy_autodiff_barrier(running_var)
    else
        N = ndims(x)
        dims = collect([1:(N - 2); N])
        rm = running_mean === nothing ? mean(x; dims) : running_mean
        rv = running_var === nothing ? var(x; mean=rm, dims, corrected=false) : running_var
    end
    return rm, rv
end

function _batchnorm_cudnn! end

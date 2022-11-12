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
function batchnorm(x::AbstractArray{<:Real, N},
                   scale::Union{AbstractVector{<:Real}, Nothing},
                   bias::Union{AbstractVector{<:Real}, Nothing},
                   running_mean::Union{AbstractVector{<:Real}, Nothing},
                   running_var::Union{AbstractVector{<:Real}, Nothing}; momentum::Real,
                   training::Val, epsilon::Real) where {N}
    x_, xm, xv = _normalization(x, running_mean, running_var, scale, bias,
                                _get_batchnorm_reduce_dims(x), training, momentum, epsilon)

    return x_, (; running_mean=xm, running_var=xv)
end

@generated function _get_batchnorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return :($(Val(Tuple(collect([1:(N - 2); N])))))
end

_CUDNN_BATCHNORM_FLOAT = Union{Float32, Float64}

_CUDNN_BATCHNORM_ARRAY_TYPE = Union{CuArray{<:_CUDNN_BATCHNORM_FLOAT, 2},
                                    CuArray{<:_CUDNN_BATCHNORM_FLOAT, 4},
                                    CuArray{<:_CUDNN_BATCHNORM_FLOAT, 5}}

function batchnorm(x::_CUDNN_BATCHNORM_ARRAY_TYPE,
                   scale::Union{CuVector{<:_CUDNN_BATCHNORM_FLOAT}, Nothing},
                   bias::Union{CuVector{<:_CUDNN_BATCHNORM_FLOAT}, Nothing},
                   running_mean::Union{CuVector{<:_CUDNN_BATCHNORM_FLOAT}, Nothing},
                   running_var::Union{CuVector{<:_CUDNN_BATCHNORM_FLOAT}, Nothing};
                   momentum::Real, training::Val, epsilon::Real)
    rm, rv = _get_batchnorm_statistics(x, running_mean, running_var, training)

    x_ = _batchnorm_cudnn!(rm, rv, scale, bias, x, momentum, epsilon, training)
    return x_, (; running_mean=rm, running_var=rv)
end

function _get_batchnorm_statistics(x, running_mean, running_var,
                                   ::Val{training}) where {training}
    if training
        # NNlibCUDA silently updates running_mean and running_var. Copying them!
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

function _batchnorm_cudnn!(running_mean, running_var, scale, bias, x, momentum, eps,
                           ::Val{training}) where {training}
    return NNlibCUDA.batchnorm(scale, bias, x, running_mean, running_var, momentum; eps,
                               training)
end

function CRC.rrule(::typeof(_batchnorm_cudnn!), running_mean, running_var, scale, bias, x,
                   momentum, epsilon, t::Val{training}) where {training}
    y = _batchnorm_cudnn!(running_mean, running_var, scale, bias, x, momentum, epsilon, t)
    function _batchnorm_cudnn!_pullback(dy)
        dg, db, dx = NNlibCUDA.∇batchnorm(scale, bias, x, unthunk(dy), running_mean,
                                          running_var, momentum; eps=epsilon, training)
        return (NoTangent(), NoTangent(), NoTangent(), dg, db, dx, NoTangent(), NoTangent(),
                NoTangent())
    end
    return y, _batchnorm_cudnn!_pullback
end

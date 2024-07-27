@doc doc"""
    batchnorm(x, scale, bias, running_mean, running_var, training, σ=identity,
        momentum = 0.1f0, epsilon = eps(eltype(x)) ^ (5 // 7))

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
  - `training`: Set to `Val(true)` if running in training mode
  - `σ`: Activation function (default: `identity`)
  - `momentum`: Momentum for updating running mean and variance (default: `0.1f0`)
  - `epsilon`: Value added to the denominator for numerical stability
    (default: `eps(eltype(x)) ^ (5 / 7)`)

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
function batchnorm(x::AbstractArray{<:Real, N}, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, running_mean::Optional{<:AbstractVector},
        running_var::Optional{<:AbstractVector}, training::Val, σ::F=identity,
        momentum::Real=0.1f0, epsilon::Real=__default_epsilon(x)) where {F, N}
    x_, xm, xv = _batchnorm_impl(
        x, __value(running_mean), __value(running_var), scale, bias,
        _get_batchnorm_reduce_dims(x), training, momentum, epsilon,
        select_fastest_activation(σ, x, scale, bias, running_mean, running_var))
    return (x_, (; running_mean=__value(xm), running_var=__value(xv)))
end

@generated function _get_batchnorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return :($(Val(Tuple(collect([1:(N - 2); N])))))
end

# Currently used only in cuDNN
function _get_batchnorm_statistics(x, running_mean, running_var, ::Val{true})
    return _copy_autodiff_barrier(running_mean), _copy_autodiff_barrier(running_var)
end

function _get_batchnorm_statistics(
        x::AbstractArray{T, N}, running_mean, running_var, ::Val{false}) where {T, N}
    dims = collect([1:(N - 2); N])
    @assert !((running_mean === nothing) ⊻ (running_var === nothing))
    running_mean === nothing && return fast_mean_var(x; dims, corrected=false)
    return running_mean, running_var
end

CRC.@non_differentiable _get_batchnorm_statistics(::Any...)

function batchnorm_cudnn end
function ∇batchnorm_cudnn end

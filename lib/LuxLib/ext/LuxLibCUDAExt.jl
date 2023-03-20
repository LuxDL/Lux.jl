module LuxLibCUDAExt

using ..LuxCUDA, ..LuxLib, ..ChainRulesCore
import ..ChainRulesCore as CRC

# utils.jl
LuxLib._replicate(rng::CUDA.RNG) = deepcopy(rng)

# impl/groupnorm.jl
LuxLib._linear_threads_groupnorm(::CUDADevice) = (16, 16)

# api/batchnorm.jl
_CUDNN_BATCHNORM_FLOAT = Union{Float32, Float64}

_CUDNN_BATCHNORM_ARRAY_TYPE = Union{CuArray{<:_CUDNN_BATCHNORM_FLOAT, 2},
                                    CuArray{<:_CUDNN_BATCHNORM_FLOAT, 4},
                                    CuArray{<:_CUDNN_BATCHNORM_FLOAT, 5}}

_CUDNN_BN_ARRAY_NOTHING = Union{CuVector{<:_CUDNN_BATCHNORM_FLOAT}, Nothing}

function LuxLib.batchnorm(x::_CUDNN_BATCHNORM_ARRAY_TYPE, scale::_CUDNN_BN_ARRAY_NOTHING,
                          bias::_CUDNN_BN_ARRAY_NOTHING, rmean::_CUDNN_BN_ARRAY_NOTHING,
                          rvar::_CUDNN_BN_ARRAY_NOTHING; momentum::Real, training::Val,
                          epsilon::Real)
    running_mean, running_var = _get_batchnorm_statistics(x, rmean, rvar, training)

    x_ = LuxLib._batchnorm_cudnn!(running_mean, running_var, scale, bias, x, momentum,
                                  epsilon, training)
    return x_, (; running_mean, running_var)
end

function _get_batchnorm_statistics(x, running_mean, running_var,
                                   ::Val{training}) where {training}
    if training
        # NNlibCUDA silently updates running_mean and running_var. Copying them!
        rm = LuxLib._copy_autodiff_barrier(running_mean)
        rv = LuxLib._copy_autodiff_barrier(running_var)
    else
        N = ndims(x)
        dims = collect([1:(N - 2); N])
        rm = running_mean === nothing ? mean(x; dims) : running_mean
        rv = running_var === nothing ? var(x; mean=rm, dims, corrected=false) : running_var
    end
    return rm, rv
end

function LuxLib._batchnorm_cudnn!(running_mean, running_var, scale, bias, x, momentum, eps,
                                  ::Val{training}) where {training}
    return NNlibCUDA.batchnorm(scale, bias, x, running_mean, running_var, momentum; eps,
                               training)
end

function CRC.rrule(::typeof(LuxLib._batchnorm_cudnn!), running_mean, running_var, scale,
                   bias, x, momentum, epsilon, t::Val{training}) where {training}
    y = LuxLib._batchnorm_cudnn!(running_mean, running_var, scale, bias, x, momentum,
                                 epsilon, t)
    function _batchnorm_cudnn!_pullback(dy)
        dg, db, dx = NNlibCUDA.∇batchnorm(scale, bias, x, unthunk(dy), running_mean,
                                          running_var, momentum; eps=epsilon, training)
        return (NoTangent(), NoTangent(), NoTangent(), dg, db, dx, NoTangent(), NoTangent(),
                NoTangent())
    end
    return y, _batchnorm_cudnn!_pullback
end

end

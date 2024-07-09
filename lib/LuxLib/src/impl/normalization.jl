# Generic Normalization Implementation
@generated function _update_normalization_statistics(
        x::AbstractArray{T, N}, rμ::AbstractArray{<:Number, N},
        rσ²::AbstractArray{<:Number, N}, μ::AbstractArray{<:Number, N},
        σ²::AbstractArray{<:Number, N}, momentum::Real,
        r::Val{reduce_dims}) where {T, N, reduce_dims}
    return quote
        m = __value($(T)(__accum_size(x, r)))
        m_ = momentum * m / (m - one(m))
        $(if last(reduce_dims) != N
            :(μ = mean(μ; dims=N);
            σ² = mean(σ²; dims=N))
        end)
        rμ = @. (1 - momentum) * rμ + momentum * μ
        rσ² = @. (1 - momentum) * rσ² + m_ * σ²
        return rμ, rσ²
    end
end

__accum_size(x, ::Val{dims}) where {dims} = prod(Base.Fix1(size, x), dims)

CRC.@non_differentiable __accum_size(::Any...)
EnzymeRules.inactive_noinl(::typeof(__accum_size), ::Any...) = nothing

function _get_batch_statistics(
        x::AbstractArray, ::Nothing, ::Nothing, ::Val{rdims}, ::Val, momentum) where {rdims}
    μ = __aos_to_soa(mean(x; dims=rdims))
    σ² = __aos_to_soa(var(x; corrected=false, mean=μ, dims=rdims))
    return (μ, σ²), (nothing, nothing)
end

function _get_batch_statistics(::AbstractArray, rμ::AbstractArray, rσ²::AbstractArray,
        ::Val{rdims}, ::Val{false}, momentum) where {rdims}
    return (rμ, rσ²), (rμ, rσ²)
end

function _get_batch_statistics(x::AbstractArray, rμ::AbstractArray, rσ²::AbstractArray,
        r::Val{rdims}, ::Val{true}, momentum) where {rdims}
    μ = __aos_to_soa(mean(x; dims=rdims))
    σ² = __aos_to_soa(var(x; corrected=false, mean=μ, dims=rdims))
    rμ, rσ² = _update_normalization_statistics(
        __value(x), __value(rμ), __value(rσ²), __value(μ), __value(σ²), momentum, r)
    return (μ, σ²), (rμ, rσ²)
end

function _normalization_impl(x::AbstractArray, running_mean::Optional{<:AbstractArray},
        running_var::Optional{<:AbstractArray}, scale::Optional{<:AbstractArray},
        bias::Optional{<:AbstractArray}, r::Val{reduce_dims}, training::Val,
        momentum, epsilon, act::F=identity) where {reduce_dims, F}
    (μ, σ²), (rμ, rσ²) = _get_batch_statistics(
        x, running_mean, running_var, r, training, momentum)
    return _affine_normalize(act, x, μ, σ², scale, bias, epsilon), rμ, rσ²
end

@stable default_mode="warn" _normalization(args...)=__normalization(args...)

function __normalization(x::AbstractArray, running_mean::Optional{<:AbstractVector},
        running_var::Optional{<:AbstractVector}, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, reduce_dims::Val,
        training::Val, momentum, epsilon, act::F=identity) where {F}
    x_, rμ, rσ² = _normalization_impl(x, _reshape_into_proper_shape(running_mean, x),
        _reshape_into_proper_shape(running_var, x), _reshape_into_proper_shape(scale, x),
        _reshape_into_proper_shape(bias, x), reduce_dims, training, momentum, epsilon, act)
    return x_, _vec(rμ), _vec(rσ²)
end

# Here we reorder the operations a bit for better performance
function _affine_normalize(::typeof(identity), x::AbstractArray, xmean,
        xvar, ::Nothing, ::Nothing, epsilon::Real)
    _scale = @. inv(sqrt(xvar + epsilon))
    _bias = @. xmean * _scale
    return @. x * _scale - _bias
end
function _affine_normalize(act::F, x::AbstractArray, xmean, xvar,
        ::Nothing, ::Nothing, epsilon::Real) where {F}
    _scale = @. inv(sqrt(xvar + epsilon))
    _bias = @. xmean * _scale
    return @. act(x * _scale - _bias)
end
function _affine_normalize(::typeof(identity), x::AbstractArray, xmean, xvar,
        scale::AbstractArray, bias::AbstractArray, epsilon::Real)
    _scale = @. scale / sqrt(xvar + epsilon)
    _bias = @. bias - xmean * _scale
    return @. x * _scale + _bias
end
function _affine_normalize(act::F, x::AbstractArray, xmean, xvar, scale::AbstractArray,
        bias::AbstractArray, epsilon::Real) where {F}
    _scale = @. scale / sqrt(xvar + epsilon)
    _bias = @. bias - xmean * _scale
    return @. act(x * _scale + _bias)
end

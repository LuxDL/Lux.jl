# Generic Normalization Implementation
@inline function _update_normalization_statistics(
        x::AbstractArray{<:Number, N}, rμ::AbstractArray{<:Number, N},
        rσ²::AbstractArray{<:Number, N}, μ::AbstractArray{<:Number, N},
        σ²::AbstractArray{<:Number, N}, momentum::Real,
        ::Val{reduce_dims}) where {N, reduce_dims}
    m = eltype(x)(prod(Base.Fix1(size, x), reduce_dims))
    m_ = m / (m - one(m))
    if last(reduce_dims) != N
        μ = mean(μ; dims=N)
        σ² = mean(σ²; dims=N)
    end
    rμ = @. (1 - momentum) * rμ + momentum * μ
    rσ² = @. (1 - momentum) * rσ² + momentum * σ² * m_
    return rμ, rσ²
end

@inline function _get_batch_statistics(x::AbstractArray, ::Nothing, ::Nothing,
        ::Val{rdims}, ::Val{false}, momentum) where {rdims}
    μ = mean(x; dims=rdims)
    σ² = var(x; corrected=false, mean=μ, dims=rdims)
    return (μ, σ²), (nothing, nothing)
end

@inline function _get_batch_statistics(
        ::AbstractArray, rμ::AbstractArray, rσ²::AbstractArray,
        ::Val{rdims}, ::Val{false}, momentum) where {rdims}
    return (rμ, rσ²), (rμ, rσ²)
end

@inline function _get_batch_statistics(x::AbstractArray, ::Nothing, ::Nothing,
        ::Val{rdims}, ::Val{true}, momentum) where {rdims}
    μ = mean(x; dims=rdims)
    σ² = var(x; corrected=false, mean=μ, dims=rdims)
    return (μ, σ²), (nothing, nothing)
end

@inline function _get_batch_statistics(
        x::AbstractArray, rμ::AbstractArray, rσ²::AbstractArray,
        r::Val{rdims}, ::Val{true}, momentum) where {rdims}
    μ = mean(x; dims=rdims)
    σ² = var(x; corrected=false, mean=μ, dims=rdims)
    rμ, rσ² = _update_normalization_statistics(x, rμ, rσ², μ, σ², momentum, r)
    return (μ, σ²), (rμ, rσ²)
end

@inline function _normalization_impl(
        x::AbstractArray, running_mean::Union{Nothing, <:AbstractArray},
        running_var::Union{Nothing, <:AbstractArray},
        scale::Union{Nothing, <:AbstractArray}, bias::Union{Nothing, <:AbstractArray},
        r::Val{reduce_dims}, training::Val, momentum,
        epsilon, act::F=identity) where {reduce_dims, F}
    (μ, σ²), (rμ, rσ²) = _get_batch_statistics(
        x, running_mean, running_var, r, training, momentum)
    return _affine_normalize(act, x, μ, σ², scale, bias, epsilon), rμ, rσ²
end

function _normalization(x::AbstractArray, running_mean::Union{Nothing, <:AbstractVector},
        running_var::Union{Nothing, <:AbstractVector},
        scale::Union{Nothing, <:AbstractVector},
        bias::Union{Nothing, <:AbstractVector}, reduce_dims::Val,
        training::Val, momentum, epsilon, act::F=identity) where {F}
    x_, rμ, rσ² = _normalization_impl(x, _reshape_into_proper_shape(running_mean, x),
        _reshape_into_proper_shape(running_var, x), _reshape_into_proper_shape(scale, x),
        _reshape_into_proper_shape(bias, x), reduce_dims, training, momentum, epsilon, act)
    return x_, _vec(rμ), _vec(rσ²)
end

function _affine_normalize(act::F, x::AbstractArray, xmean::AbstractArray,
        xvar::AbstractArray, ::Nothing, ::Nothing, epsilon::Real) where {F}
    act === identity && return @. (x .- xmean) / sqrt(xvar + epsilon)
    return @. act((x .- xmean) / sqrt(xvar + epsilon))
end

function _affine_normalize(
        act::F, x::AbstractArray, xmean::AbstractArray, xvar::AbstractArray,
        scale::AbstractArray, bias::AbstractArray, epsilon::Real) where {F}
    # Here we reorder the operations a bit for better performance
    _scale = @. scale / sqrt(xvar + epsilon)
    _bias = @. bias - xmean * _scale
    act === identity && return @. x * _scale + _bias
    return @. act(x * _scale + _bias)
end

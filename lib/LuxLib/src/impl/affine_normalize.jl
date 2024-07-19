@stable default_mode="warn" function _affine_normalize(
        f::F, x::AbstractArray, xmean, xvar, scale, bias, epsilon::Real) where {F}
    return __affine_normalize(f, x, xmean, xvar, scale, bias, epsilon)
end

function __affine_normalize(::typeof(identity), x::AbstractArray, xmean,
        xvar, ::Nothing, ::Nothing, epsilon::Real)
    _scale = @. inv(sqrt(xvar + epsilon))
    _bias = @. xmean * _scale
    return @. x * _scale - _bias
end

function __affine_normalize(act::F, x::AbstractArray, xmean, xvar,
        ::Nothing, ::Nothing, epsilon::Real) where {F}
    _scale = @. inv(sqrt(xvar + epsilon))
    _bias = @. xmean * _scale
    return @. act(x * _scale - _bias)
end

function __affine_normalize(::typeof(identity), x::AbstractArray, xmean, xvar,
        scale::AbstractArray, bias::AbstractArray, epsilon::Real)
    _scale = @. scale / sqrt(xvar + epsilon)
    _bias = @. bias - xmean * _scale
    return @. x * _scale + _bias
end

function __affine_normalize(act::F, x::AbstractArray, xmean, xvar, scale::AbstractArray,
        bias::AbstractArray, epsilon::Real) where {F}
    _scale = @. scale / sqrt(xvar + epsilon)
    _bias = @. bias - xmean * _scale
    return @. act(x * _scale + _bias)
end

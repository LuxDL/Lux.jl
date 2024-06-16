"""
    xlogx(x::Number)

Return `x * log(x)` for `x ≥ 0`, handling `x == 0` by taking the limit from above, to get
zero.
"""
@inline function xlogx(x::Number)
    result = x * log(x)
    return ifelse(iszero(x), zero(result), result)
end

function CRC.rrule(::typeof(xlogx), x::Number)
    iszero(x) && return x, Δ -> (NoTangent(), ZeroTangent())
    logx = log(x)
    ∇xlogx = @closure Δ -> (NoTangent(), @thunk(Δ*(logx + true)))
    return x * logx, ∇xlogx
end

function CRC.rrule(
        ::typeof(Broadcast.broadcasted), ::typeof(xlogx), x::AbstractArray{<:Number})
    logx = log.(x)
    y = x .* logx
    ∇xlogx = @closure Δ -> (NoTangent(), NoTangent(), @thunk(Δ.*(logx .+ true)))
    return y, ∇xlogx
end

"""
    xlogy(x::Number, y::Number)

Return `x * log(y)` for `y > 0`, and zero when `x == 0`.
"""
@inline function xlogy(x::Number, y::Number)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function CRC.rrule(::typeof(xlogy), x::Number, y::Number)
    iszero(x) && return x, Δ -> (NoTangent(), ZeroTangent())
    logy = log(y)
    ∇xlogy = @closure Δ -> (NoTangent(), @thunk(Δ*logy), @thunk(Δ * x/y))
    return x * logy, ∇xlogy
end

function CRC.rrule(::typeof(Broadcast.broadcasted), ::typeof(xlogy),
        x::AbstractArray{<:Number}, y::AbstractArray{<:Number})
    logy = log.(y)
    y = x .* logy
    ∇xlogy = @closure Δ -> (NoTangent(), NoTangent(), @thunk(Δ.*logy), @thunk(Δ .* x./y))
    return y, ∇xlogy
end

@inline function __check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y))
        if size(ŷ, d) != size(y, d)
            throw(DimensionMismatch("loss function expects size(ŷ) = $(size(ŷ)) to match \
                                     size(y) = $(size(y))"))
        end
    end
end
@inline __check_sizes(ŷ, y) = nothing

CRC.@non_differentiable __check_sizes(ŷ::Any, y::Any)

@inline __fused_agg(::typeof(mean), op::OP, x) where {OP} = mean(op, x)
@inline __fused_agg(::typeof(sum), op::OP, x) where {OP} = sum(op, x)
@inline __fused_agg(::Nothing, op::OP, x) where {OP} = op.(x)
@inline __fused_agg(f::F, op::OP, x) where {F, OP} = f(op.(x))

@inline __label_smoothing(::Nothing, y, ::Type{T}) where {T} = y
@inline function __label_smoothing(label_smoothing::Real, y, ::Type{T}) where {T}
    label_smoothing = T(label_smoothing)
    return y .* (1 - label_smoothing) .+ label_smoothing ./ size(y, ndims(y) - 1)
end

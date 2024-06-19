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

# Some functional forms of losses

@inline function __siamese_contrastive_loss(x::T1, y::T2, margin=true) where {T1, T2}
    return (1 - y) * x^2 + y * max(promote_type(T1, T2)(0), margin - x)^2
end

@inline function __poisson_loss(x::T1, y::T2, ϵ) where {T1, T2}
    return x - xlogy(y, x + __get_epsilon(T1, ϵ))
end

@inline function __msle_loss(x::T1, y::T2, ϵ) where {T1, T2}
    ϵ = __get_epsilon(promote_type(T1, T2), ϵ)
    return log((x + ϵ) / (y + ϵ))^2
end

# Misc Utils

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

@inline function __fused_agg(::typeof(mean), op::OP, x) where {OP}
    return __fused_agg(sum, op, x) / length(x)
end
@inline function __fused_agg(::typeof(mean), lfn::LossFunctions.Traits.Loss, x, y)
    return __fused_agg(sum, lfn, x, y) / length(x)
end

@inline __fused_agg(::typeof(sum), op::OP, x) where {OP} = sum(op, x)
@inline function __fused_agg(::typeof(sum), lfn::LossFunctions.Traits.Loss, x, y)
    fast_scalar_indexing(x) && fast_scalar_indexing(y) && return sum(lfn, x, y)
    return mapreduce(Broadcast.BroadcastFunction(lfn), +, x, y)
end

@inline function CRC.rrule(
        ::typeof(__fused_agg), ::typeof(sum), lfn::LossFunctions.Traits.Loss, x, y)
    z = lfn.(x, y)
    ∇lfn = let z = z, y = y, lfn = lfn
        Δ -> begin
            ∂x = @thunk LossFunctions.deriv.((lfn,), z, y) .* Δ
            return NoTangent(), NoTangent(), NoTangent(), ∂x, NoTangent()
        end
    end
    return sum(z), ∇lfn
end

@inline __fused_agg(::Nothing, op::OP, args...) where {OP} = op.(args...)
@inline __fused_agg(f::F, op::OP, args...) where {F, OP} = f(op.(args...))

@inline __label_smoothing(::Nothing, y, ::Type{T}) where {T} = y
@inline function __label_smoothing(label_smoothing::Real, y, ::Type{T}) where {T}
    label_smoothing = T(label_smoothing)
    return y .* (1 - label_smoothing) .+ label_smoothing ./ size(y, ndims(y) - 1)
end

@inline __label_smoothing_binary(::Nothing, y, ::Type{T}) where {T} = y
@inline function __label_smoothing_binary(label_smoothing::Real, y, ::Type{T}) where {T}
    label_smoothing = T(label_smoothing)
    return y .* (1 - label_smoothing) .+ label_smoothing ./ 2
end

@inline __get_epsilon(::Type{T}, ϵ::Real) where {T} = T(ϵ)
@inline __get_epsilon(::Type{T}, ::Nothing) where {T} = eps(float(T))

@inline __get_dims(_) = Colon()
@inline __get_dims(::AbstractVector) = Colon()
@inline __get_dims(::AbstractArray{T, N}) where {T, N} = 1:(N - 1)

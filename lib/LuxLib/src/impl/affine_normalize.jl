# This is the generic implementation. Helpful because we don't need to manually reshape
# arrays and such.
function _affine_normalize(
        act::F, x::AbstractArray, μ, σ², ::Nothing, ::Nothing, ϵ::Real) where {F}
    _scale = @. inv(sqrt(σ² + ϵ))
    _bias = @. μ * _scale
    return @. act(x * _scale - _bias)
end

function _affine_normalize(act::F, x::AbstractArray, μ, σ², scale::AbstractArray,
        bias::AbstractArray, ϵ::Real) where {F}
    _scale = @. scale / sqrt(σ² + ϵ)
    _bias = @. bias - μ * _scale
    return @. act(x * _scale + _bias)
end

# Specialized affine normalize that is generally faster that the above generic
# implementation. We bypass julia's broadcasting mechanism if we can. We still might fall
# back to the generic implementation if we must (like for ForwardDiff/Tracker/ReverseDiff)

## Group Normalization

function _affine_normalize_gn(
        f::F, x::AbstractArray, μ, σ², scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, ϵ::Real) where {F}
    return _affine_normalize_gn(
        internal_operation_mode((x, μ, σ², scale, bias)), f, x, μ, σ², scale, bias, ϵ)
end

function _affine_normalize_gn(::GenericBroadcastOp, f::F, x::AbstractArray,
        μ, σ², scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, ϵ::Real) where {F}
    return _affine_normalize(f, x, μ, σ², _reshape_into_normalization_shape(scale, x),
        _reshape_into_normalization_shape(bias, x), ϵ)
end

function _affine_normalize_gn(opmode::AbstractInternalArrayOpMode, f::F,
        x::AbstractArray{T, N}, μ, σ², scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, ϵ::Real) where {F, T, N}
    x_ = reshape(x, :, size(x, N - 2), size(x, N - 1), size(x, N))
    μ_ = reshape(μ, 1, 1, size(x, N - 1), size(x, N))
    σ²_ = reshape(σ², 1, 1, size(x, N - 1), size(x, N))
    scale_ = __reshape(scale, 1, size(x, N - 2), size(x, N - 1), 1)
    bias_ = __reshape(bias, 1, size(x, N - 2), size(x, N - 1), 1)

    return _affine_normalize_gn_impl(opmode, f, x_, μ_, σ²_, scale_, bias_, ϵ)
end

function _affine_normalize_gn_impl(opmode::AbstractInternalArrayOpMode, f::F,
        x::AbstractArray{T, N}, μ, σ², scale::Optional{<:AbstractArray},
        bias::Optional{<:AbstractArray}, ϵ::Real) where {F, T, N}
    y = similar(x,
        promote_type(
            __eltype(x), __eltype(μ), __eltype(σ²), __eltype(scale), __eltype(bias)))
    __affine_normalize_gn_impl!(opmode, y, f, x, μ, σ², scale, bias, ϵ)
    return y
end

function __affine_normalize_gn_impl!(::LoopedArrayOp, y::AbstractArray{<:Number, 4}, f::F,
        x::AbstractArray{<:Number, 4}, μ, σ², ::Nothing, ::Nothing, ϵ::Real) where {F}
    for L in axes(y, 4), K in axes(y, 3)
        @inbounds _sc = @fastmath inv(sqrt(σ²[1, 1, K, L] + ϵ))
        @inbounds _bc = -μ[1, 1, K, L] * _sc
        for J in axes(y, 2)
            @simd ivdep for I in axes(y, 1)
                @inbounds y[I, J, K, L] = muladd(x[I, J, K, L], _sc, _bc)
            end
        end
    end
    _fast_activation!(f, y) # NOTE: don't fuse into the above loop
end

function __affine_normalize_gn_impl!(::LoopedArrayOp, y::AbstractArray{<:Number, 4}, f::F,
        x::AbstractArray{<:Number, 4}, μ, σ², scale::AbstractArray{<:Number, 4},
        bias::AbstractArray{<:Number, 4}, ϵ::Real) where {F}
    for L in axes(y, 4), K in axes(y, 3)
        @inbounds idenom = @fastmath inv(sqrt(σ²[1, 1, K, L] + ϵ))
        for J in axes(y, 2)
            @inbounds _sc = scale[1, J, K, 1] * idenom
            @inbounds _bc = muladd(-μ[1, 1, K, L], _sc, bias[1, J, K, 1])
            @simd ivdep for I in axes(y, 1)
                @inbounds y[I, J, K, L] = muladd(x[I, J, K, L], _sc, _bc)
            end
        end
    end
    _fast_activation!(f, y) # NOTE: don't fuse into the above loop
end

function __affine_normalize_gn_impl!(::GPUBroadcastOp, y::AbstractArray{<:Number, 4}, f::F,
        x::AbstractArray{<:Number, 4}, μ, σ², scale::Optional{<:AbstractArray},
        bias::Optional{<:AbstractArray}, ϵ::Real) where {F}
    backend = KA.get_backend(y)
    kernel! = __affine_normalize_gn_kernel!(backend)
    kernel!(y, f, x, μ, σ², scale, bias, ϵ; ndrange=size(y))
    KA.synchronize(backend)
end

@kernel function __affine_normalize_gn_kernel!(
        y::AbstractArray{<:Number, 4}, @Const(f), @Const(x),
        @Const(μ), @Const(σ²), @Const(scale), @Const(bias), @Const(ϵ))
    (i, j, k, l) = @index(Global, NTuple)
    if scale !== nothing
        @inbounds _sc = scale[1, j, k, 1] / sqrt(σ²[1, 1, k, l] + ϵ)
        @inbounds _bc = bias[1, j, k, 1] - μ[1, 1, k, l] * _sc
    else
        @inbounds _sc = inv(sqrt(σ²[1, 1, k, l] + ϵ))
        @inbounds _bc = -μ[1, 1, k, l] * _sc
    end
    @inbounds y[i, j, k, l] = f(muladd(x[i, j, k, l], _sc, _bc))
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(_affine_normalize_gn_impl),
        opmode::AbstractInternalArrayOpMode, f::F,
        x::AbstractArray{T, N}, μ, σ², scale::Optional{<:AbstractArray},
        bias::Optional{<:AbstractArray}, ϵ::Real) where {F, T, N}
    y = similar(x,
        promote_type(
            __eltype(x), __eltype(μ), __eltype(σ²), __eltype(scale), __eltype(bias)))
    __affine_normalize_gn_impl!(opmode, y, identity, x, μ, σ², scale, bias, ϵ)
    z, ∇activation = CRC.rrule_via_ad(cfg, fast_activation!!, f, y)

    proj_x = CRC.ProjectTo(x)
    proj_μ = CRC.ProjectTo(μ)
    proj_σ² = CRC.ProjectTo(σ²)
    proj_sc = scale === nothing ? identity : CRC.ProjectTo(scale)
    proj_bi = bias === nothing ? identity : CRC.ProjectTo(bias)

    ∇affine_normalize_gn_impl_internal = @closure Δ -> begin
        ∂y = last(∇activation(Δ))
        ∂x, ∂μ, ∂σ², ∂sc, ∂b = ∇affine_normalize_gn_impl(
            opmode, ∂y, x, μ, σ², scale, bias, ϵ)
        return (
            ∂∅, ∂∅, ∂∅, proj_x(∂x), proj_μ(∂μ), proj_σ²(∂σ²), proj_sc(∂sc), proj_bi(∂b), ∂∅)
    end

    return z, ∇affine_normalize_gn_impl_internal
end

# NOTE: Technically we can cache intermediate results in the forward pass. But that might
#       not lead to much speedup.

function ∇affine_normalize_gn_impl(::GPUBroadcastOp, ∂y, x, μ, σ², scale, bias, ϵ)
    ∂x = similar(x)
    ∂μ = similar(μ, size(x))
    ∂σ² = similar(σ², size(x))
    ∂sc = scale === nothing ? ∂∅ : similar(scale, size(x))
    ∂b = bias === nothing ? ∂∅ : similar(bias, size(x))

    backend = KA.get_backend(∂x)
    kernel! = ∇affine_normalize_gn_kernel!(backend)
    kernel!(∂x, ∂μ, ∂σ², ∂sc, ∂b, ∂y, x, μ, σ², scale, bias, ϵ; ndrange=size(∂x))
    KA.synchronize(backend)

    return (∂x, __reduce_sum(μ, ∂μ), __reduce_sum(σ², ∂σ²),
        __reduce_sum(scale, ∂sc), __reduce_sum(bias, ∂b))
end

@kernel function ∇affine_normalize_gn_kernel!(
        ∂x, ∂μ, ∂σ², ∂sc, ∂b, @Const(∂y), @Const(x), @Const(μ),
        @Const(σ²), @Const(scale), @Const(bias), @Const(ϵ))
    (i, j, k, l) = @index(Global, NTuple)
    @inbounds denom = sqrt(σ²[1, 1, k, l] + ϵ)
    @inbounds denom² = denom * denom
    if scale !== nothing
        @inbounds _sc = scale[1, j, k, 1] / denom
    else
        @inbounds _sc = inv(denom)
    end
    @inbounds xμ = x[i, j, k, l] - μ[1, 1, k, l]

    @inbounds ∂x[i, j, k, l] = ∂y[i, j, k, l] * _sc
    @inbounds ∂μ[i, j, k, l] = -∂x[i, j, k, l]
    @inbounds ∂σ²[i, j, k, l] = -∂x[i, j, k, l] * xμ / (2 * denom²)

    if scale !== nothing
        @inbounds ∂sc[i, j, k, l] = ∂y[i, j, k, l] * xμ / denom
        @inbounds ∂b[i, j, k, l] = ∂y[i, j, k, l]
    end
end

function ∇affine_normalize_gn_impl(::LoopedArrayOp, ∂y, x, μ, σ², ::Nothing, ::Nothing, ϵ)
    ∂x, ∂μ, ∂σ² = similar(x), zero.(μ), zero.(σ²)
    half = eltype(∂σ²)(0.5)

    for L in axes(∂y, 4), K in axes(∂y, 3)
        @inbounds idenom = @fastmath inv(sqrt(σ²[1, 1, K, L] + ϵ))
        idenom² = idenom^2
        for J in axes(∂y, 2)
            @simd for I in axes(∂y, 1)
                @inbounds xμ = x[I, J, K, L] - μ[1, 1, K, L]

                @inbounds ∂x[I, J, K, L] = ∂y[I, J, K, L] * idenom
                @inbounds ∂μ[1, 1, K, L] -= ∂x[I, J, K, L]
                @inbounds ∂σ²[1, 1, K, L] -= ∂x[I, J, K, L] * xμ * half * idenom²
            end
        end
    end

    return ∂x, ∂μ, ∂σ², ∂∅, ∂∅
end

function ∇affine_normalize_gn_impl(::LoopedArrayOp, ∂y, x, μ, σ², scale, bias, ϵ)
    ∂x, ∂μ, ∂σ², ∂sc, ∂b = similar(x), zero.(μ), zero.(σ²), zero.(scale), zero.(bias)
    half = eltype(∂σ²)(0.5)

    for L in axes(∂y, 4), K in axes(∂y, 3)
        @inbounds idenom = @fastmath inv(sqrt(σ²[1, 1, K, L] + ϵ))
        idenom² = idenom^2
        for J in axes(∂y, 2)
            @inbounds _sc = scale[1, J, K, 1] * idenom
            @simd for I in axes(∂y, 1)
                @inbounds xμ = x[I, J, K, L] - μ[1, 1, K, L]

                @inbounds ∂x[I, J, K, L] = ∂y[I, J, K, L] * _sc
                @inbounds ∂μ[1, 1, K, L] -= ∂x[I, J, K, L]
                @inbounds ∂σ²[1, 1, K, L] -= ∂x[I, J, K, L] * xμ * half * idenom²
                @inbounds ∂sc[1, J, K, 1] += ∂y[I, J, K, L] * xμ * idenom
                @inbounds ∂b[1, J, K, 1] += ∂y[I, J, K, L]
            end
        end
    end

    return ∂x, ∂μ, ∂σ², ∂sc, ∂b
end

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

for norm_op in (:bn, :gn)
    op = Symbol("_affine_normalize_$(norm_op)")
    impl_op = Symbol("_affine_normalize_$(norm_op)_impl")
    impl_op! = Symbol("__affine_normalize_$(norm_op)_impl!")
    @eval begin
        function $(op)(act::F, x::AbstractArray, μ, σ², scale::Optional{<:AbstractVector},
                bias::Optional{<:AbstractVector}, ϵ::Real) where {F}
            return $(op)(internal_operation_mode((x, μ, σ², scale, bias)),
                act, x, μ, σ², scale, bias, ϵ)
        end

        function $(op)(::GenericBroadcastOp, act::F, x::AbstractArray{T, N},
                μ, σ², scale::Optional{<:AbstractVector},
                bias::Optional{<:AbstractVector}, ϵ::Real) where {F, T, N}
            return _affine_normalize(
                act, x, μ, σ², _reshape_into_normalization_shape(scale, x),
                _reshape_into_normalization_shape(bias, x), ϵ)
        end

        function $(impl_op)(opmode::AbstractInternalArrayOpMode, act::F,
                x::AbstractArray{T, N}, μ, σ², scale::Optional{<:AbstractArray},
                bias::Optional{<:AbstractArray}, ϵ::Real) where {F, T, N}
            y = similar(x,
                promote_type(__eltype(x), __eltype(μ), __eltype(σ²),
                    __eltype(scale), __eltype(bias)))
            $(impl_op!)(opmode, y, act, x, μ, σ², scale, bias, ϵ)
            return y
        end
    end
end

## Batch Normalization

function _affine_normalize_bn(opmode::AbstractInternalArrayOpMode, f::F,
        x::AbstractArray{T, N}, μ, σ², scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, ϵ::Real) where {F, T, N}
    x_ = reshape(x, :, size(x, N - 1), size(x, N))
    μ_ = reshape(μ, 1, size(x, N - 1), 1)
    σ²_ = reshape(σ², 1, size(x, N - 1), 1)
    scale_ = __reshape(scale, 1, size(x, N - 1), 1)
    bias_ = __reshape(bias, 1, size(x, N - 1), 1)

    return reshape(
        _affine_normalize_bn_impl(opmode, f, x_, μ_, σ²_, scale_, bias_, ϵ), size(x))
end

function __affine_normalize_bn_impl!(
        ::LoopedArrayOp, y::AbstractArray{<:Number, 3}, f::F, x::AbstractArray{<:Number, 3},
        μ, σ², scale::Optional{<:AbstractArray{<:Number, 3}},
        bias::Optional{<:AbstractArray{<:Number, 3}},
        ϵ::Real, _sc::Optional{<:AbstractVector}=nothing,
        _bc::Optional{<:AbstractVector}=nothing) where {F}
    N = size(y, 2)
    _scale = _sc === nothing ?
             similar(x, promote_type(__eltype(scale), __eltype(σ²), __eltype(ϵ)), N) : _sc
    _bias = _bc === nothing ?
            similar(x, promote_type(__eltype(bias), __eltype(_scale), __eltype(ϵ)), N) : _bc

    if scale !== nothing
        @tturbo for J in indices((_scale, scale, σ², _bias, μ, bias), (1, 2, 2, 1, 2, 2))
            _scale[J] = scale[1, J, 1] / sqrt(σ²[1, J, 1] + ϵ)
            _bias[J] = -μ[1, J, 1] * _scale[J] + bias[1, J, 1]
        end
    else
        @tturbo for J in indices((_scale, σ², μ, _bias), (1, 2, 2, 1))
            _scale[J] = inv(sqrt(σ²[1, J, 1] + ϵ))
            _bias[J] = -μ[1, J, 1] * _scale[J]
        end
    end

    @tturbo for K in indices((x, y), 3),
        J in indices((x, y, _scale, _bias), (2, 2, 1, 1)),
        I in indices((x, y), 1)

        y[I, J, K] = x[I, J, K] * _scale[J] + _bias[J]
    end
    _fast_activation!(f, y) # NOTE: don't fuse into the above loop
end

function __affine_normalize_bn_impl!(::GPUBroadcastOp, y::AbstractArray{<:Number, 3},
        f::F, x::AbstractArray{<:Number, 3}, μ, σ²,
        scale::Optional{<:AbstractArray{<:Number, 3}},
        bias::Optional{<:AbstractArray{<:Number, 3}},
        ϵ::Real, _sc::Optional{<:AbstractVector}=nothing,
        _bc::Optional{<:AbstractVector}=nothing) where {F}
    backend = KA.get_backend(y)
    if _sc === nothing
        kernel! = __affine_normalize_bn_kernel!(backend)
        kernel!(y, f, x, μ, σ², scale, bias, ϵ; ndrange=size(y))
    else
        kernel! = __affine_normalize_bn_kernel_cached!(backend)
        kernel!(y, _sc, _bc, f, x, μ, σ², scale, bias, ϵ; ndrange=size(y))
    end
    KA.synchronize(backend)
end

@kernel function __affine_normalize_bn_kernel!(
        y::AbstractArray{<:Number, 3}, @Const(f), @Const(x),
        @Const(μ), @Const(σ²), @Const(scale), @Const(bias), @Const(ϵ))
    (i, j, k) = @index(Global, NTuple)
    if scale !== nothing
        @inbounds _sc = scale[1, j, 1] / sqrt(σ²[1, j, 1] + ϵ)
        @inbounds _bc = muladd(-μ[1, j, 1], _sc, bias[1, j, 1])
    else
        @inbounds _sc = inv(sqrt(σ²[1, j, 1] + ϵ))
        @inbounds _bc = -μ[1, j, 1] * _sc
    end
    @inbounds y[i, j, k] = f(muladd(x[i, j, k], _sc, _bc))
end

@kernel function __affine_normalize_bn_kernel_cached!(
        y::AbstractArray{<:Number, 3}, _sc::AbstractArray{<:Number, 3},
        _bc::AbstractArray{<:Number, 3}, @Const(f), @Const(x),
        @Const(μ), @Const(σ²), @Const(scale), @Const(bias), @Const(ϵ))
    (i, j, k) = @index(Global, NTuple)
    if scale !== nothing
        @inbounds _sc[j] = scale[1, j, 1] / sqrt(σ²[1, j, 1] + ϵ)
        @inbounds _bc[j] = muladd(-μ[1, j, 1], _sc[1, j, 1], bias[1, j, 1])
    else
        @inbounds _sc[j] = inv(sqrt(σ²[1, j, 1] + ϵ))
        @inbounds _bc[j] = -μ[1, j, 1] * _sc[1, j, 1]
    end
    @inbounds y[i, j, k] = f(muladd(x[i, j, k], _sc[1, j, 1], _bc[1, j, 1]))
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(_affine_normalize_bn_impl),
        opmode::AbstractInternalArrayOpMode, f::F,
        x::AbstractArray{T, N}, μ, σ², scale::Optional{<:AbstractArray},
        bias::Optional{<:AbstractArray}, ϵ::Real) where {F, T, N}
    y = similar(x,
        promote_type(
            __eltype(x), __eltype(μ), __eltype(σ²), __eltype(scale), __eltype(bias)))
    _sc = similar(
        x, promote_type(__eltype(scale), __eltype(σ²), __eltype(ϵ)), size(x, N - 1))
    _bc = similar(
        x, promote_type(__eltype(bias), __eltype(_sc), __eltype(ϵ)), size(x, N - 1))
    __affine_normalize_bn_impl!(opmode, y, identity, x, μ, σ², scale, bias, ϵ, _sc, _bc)
    z, ∇activation = CRC.rrule_via_ad(cfg, fast_activation!!, f, y)

    proj_x = CRC.ProjectTo(x)
    proj_μ = CRC.ProjectTo(μ)
    proj_σ² = CRC.ProjectTo(σ²)
    proj_sc = scale === nothing ? identity : CRC.ProjectTo(scale)
    proj_bi = bias === nothing ? identity : CRC.ProjectTo(bias)

    ∇affine_normalize_bn_impl_internal = @closure Δ -> begin
        ∂y = last(∇activation(Δ))
        ∂x, ∂μ, ∂σ², ∂sc, ∂b = ∇affine_normalize_bn_impl(
            opmode, ∂y, x, μ, σ², scale, bias, ϵ, _sc)
        return (
            ∂∅, ∂∅, ∂∅, proj_x(∂x), proj_μ(∂μ), proj_σ²(∂σ²), proj_sc(∂sc), proj_bi(∂b), ∂∅)
    end

    return z, ∇affine_normalize_bn_impl_internal
end

function ∇affine_normalize_bn_impl(::GPUBroadcastOp, ∂y, x, μ, σ², scale, bias, ϵ, _sc)
    ∂x = similar(x)
    ∂μ = similar(μ, size(x))
    ∂σ² = similar(σ², size(x))
    ∂sc = scale === nothing ? ∂∅ : similar(scale, size(x))
    ∂b = bias === nothing ? ∂∅ : similar(bias, size(x))

    fill!(∂μ, false)
    fill!(∂σ², false)
    scale === nothing || fill!(∂sc, false)
    bias === nothing || fill!(∂b, false)

    backend = KA.get_backend(∂x)
    kernel! = ∇affine_normalize_bn_kernel!(backend)
    kernel!(∂x, ∂μ, ∂σ², ∂sc, ∂b, ∂y, x, μ, σ², scale, bias, ϵ, _sc; ndrange=size(∂x))
    KA.synchronize(backend)

    ∂μ_ = __reduce_sum(μ, ∂μ)
    ∂σ²_ = __reduce_sum(σ², ∂σ²)
    ∂sc_ = __reduce_sum(scale, ∂sc)
    ∂b_ = __reduce_sum(bias, ∂b)

    __unsafe_free!(∂μ)
    __unsafe_free!(∂σ²)
    __unsafe_free!(∂sc)
    __unsafe_free!(∂b)

    return ∂x, ∂μ_, ∂σ²_, ∂sc_, ∂b_
end

@kernel function ∇affine_normalize_bn_kernel!(
        ∂x, ∂μ, ∂σ², ∂sc, ∂b, @Const(∂y), @Const(x), @Const(μ),
        @Const(σ²), @Const(scale), @Const(bias), @Const(ϵ), @Const(_sc))
    (i, j, k) = @index(Global, NTuple)
    if scale !== nothing
        @inbounds idenom = inv(sqrt(σ²[1, j, 1] + ϵ))
    else
        @inbounds idenom = _sc[j]
    end
    idenom² = idenom^2

    @inbounds xμ = x[i, j, k] - μ[1, j, 1]

    @inbounds ∂x[i, j, k] = ∂y[i, j, k] * _sc[j]
    @inbounds ∂μ[i, j, k] = -∂x[i, j, k]
    @inbounds ∂σ²[i, j, k] = -∂x[i, j, k] * xμ * idenom² / 2

    if scale !== nothing
        @inbounds ∂sc[i, j, k] = ∂y[i, j, k] * xμ * idenom
        @inbounds ∂b[i, j, k] = ∂y[i, j, k]
    end
end

function ∇affine_normalize_bn_impl(
        ::LoopedArrayOp, ∂y, x, μ, σ², ::Nothing, ::Nothing, ϵ, _sc)
    ∂x, ∂μ, ∂σ² = similar(x), zero.(μ), zero.(σ²)
    half = eltype(∂σ²)(0.5)

    @tturbo for K in indices(∂y, 3), J in indices(∂y, 2)
        idenom = _sc[1, J, 1]
        idenom² = idenom^2

        for I in indices(∂y, 1)
            xμ = x[I, J, K] - μ[1, J, 1]

            ∂x[I, J, K] = ∂y[I, J, K] * idenom
            ∂μ[1, J, 1] -= ∂x[I, J, K]
            ∂σ²[1, J, 1] -= ∂x[I, J, K] * xμ * half * idenom²
        end
    end

    return ∂x, ∂μ, ∂σ², ∂∅, ∂∅
end

function ∇affine_normalize_bn_impl(::LoopedArrayOp, ∂y, x, μ, σ², scale, bias, ϵ, _sc)
    ∂x, ∂μ, ∂σ², ∂sc, ∂b = similar(x), zero.(μ), zero.(σ²), zero.(scale), zero.(bias)
    half = eltype(∂σ²)(0.5)

    @tturbo for K in indices(∂y, 3), J in indices(∂y, 2)
        idenom = inv(sqrt(σ²[1, J, 1] + ϵ))
        idenom² = idenom^2

        for I in indices(∂y, 1)
            xμ = x[I, J, K] - μ[1, J, 1]

            ∂x[I, J, K] = ∂y[I, J, K] * _sc[1, J, 1]
            ∂μ[1, J, 1] -= ∂x[I, J, K]
            ∂σ²[1, J, 1] -= ∂x[I, J, K] * xμ * half * idenom²
            ∂sc[1, J, 1] += ∂y[I, J, K] * xμ * idenom
            ∂b[1, J, 1] += ∂y[I, J, K]
        end
    end

    return ∂x, ∂μ, ∂σ², ∂sc, ∂b
end

## Group Normalization

function _affine_normalize_gn(opmode::AbstractInternalArrayOpMode, f::F,
        x::AbstractArray{T, N}, μ, σ², scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, ϵ::Real) where {F, T, N}
    x_ = reshape(x, :, size(x, N - 2), size(x, N - 1), size(x, N))
    μ_ = reshape(μ, 1, 1, size(x, N - 1), size(x, N))
    σ²_ = reshape(σ², 1, 1, size(x, N - 1), size(x, N))
    scale_ = __reshape(scale, 1, size(x, N - 2), size(x, N - 1), 1)
    bias_ = __reshape(bias, 1, size(x, N - 2), size(x, N - 1), 1)

    return reshape(
        _affine_normalize_gn_impl(opmode, f, x_, μ_, σ²_, scale_, bias_, ϵ), size(x))
end

function __affine_normalize_gn_impl!(::LoopedArrayOp, y::AbstractArray{<:Number, 4}, f::F,
        x::AbstractArray{<:Number, 4}, μ, σ², ::Nothing, ::Nothing, ϵ::Real) where {F}
    @tturbo for L in indices(y, 4), K in indices(y, 3)
        _sc = inv(sqrt(σ²[1, 1, K, L] + ϵ))
        _bc = -μ[1, 1, K, L] * _sc
        for J in indices(y, 2), I in indices(y, 1)
            y[I, J, K, L] = muladd(x[I, J, K, L], _sc, _bc)
        end
    end
    _fast_activation!(f, y) # NOTE: don't fuse into the above loop
end

function __affine_normalize_gn_impl!(::LoopedArrayOp, y::AbstractArray{<:Number, 4}, f::F,
        x::AbstractArray{<:Number, 4}, μ, σ², scale::AbstractArray{<:Number, 4},
        bias::AbstractArray{<:Number, 4}, ϵ::Real) where {F}
    @tturbo for L in indices(y, 4), K in indices(y, 3)
        idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
        for J in indices(y, 2)
            _sc = scale[1, J, K, 1] * idenom
            _bc = muladd(-μ[1, 1, K, L], _sc, bias[1, J, K, 1])
            for I in indices(y, 1)
                y[I, J, K, L] = muladd(x[I, J, K, L], _sc, _bc)
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

    fill!(∂μ, false)
    fill!(∂σ², false)
    scale === nothing || fill!(∂sc, false)
    bias === nothing || fill!(∂b, false)

    backend = KA.get_backend(∂x)
    kernel! = ∇affine_normalize_gn_kernel!(backend)
    kernel!(∂x, ∂μ, ∂σ², ∂sc, ∂b, ∂y, x, μ, σ², scale, bias, ϵ; ndrange=size(∂x))
    KA.synchronize(backend)

    ∂μ_ = __reduce_sum(μ, ∂μ)
    ∂σ²_ = __reduce_sum(σ², ∂σ²)
    ∂sc_ = __reduce_sum(scale, ∂sc)
    ∂b_ = __reduce_sum(bias, ∂b)

    __unsafe_free!(∂μ)
    __unsafe_free!(∂σ²)
    __unsafe_free!(∂sc)
    __unsafe_free!(∂b)

    return ∂x, ∂μ_, ∂σ²_, ∂sc_, ∂b_
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

    @tturbo for L in indices(∂y, 4), K in indices(∂y, 3)
        idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
        idenom² = idenom^2

        for J in indices(∂y, 2), I in indices(∂y, 1)
            xμ = x[I, J, K, L] - μ[1, 1, K, L]

            ∂x[I, J, K, L] = ∂y[I, J, K, L] * idenom
            ∂μ[1, 1, K, L] -= ∂x[I, J, K, L]
            ∂σ²[1, 1, K, L] -= ∂x[I, J, K, L] * xμ * half * idenom²
        end
    end

    return ∂x, ∂μ, ∂σ², ∂∅, ∂∅
end

function ∇affine_normalize_gn_impl(::LoopedArrayOp, ∂y, x, μ, σ², scale, bias, ϵ)
    ∂x, ∂μ, ∂σ², ∂sc, ∂b = similar(x), zero.(μ), zero.(σ²), zero.(scale), zero.(bias)
    half = eltype(∂σ²)(0.5)

    @tturbo for L in indices(∂y, 4), K in indices(∂y, 3)
        idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
        idenom² = idenom^2

        for J in indices(∂y, 2), I in indices(∂y, 1)
            xμ = x[I, J, K, L] - μ[1, 1, K, L]

            ∂x[I, J, K, L] = ∂y[I, J, K, L] * _sc
            ∂μ[1, 1, K, L] -= ∂x[I, J, K, L]
            ∂σ²[1, 1, K, L] -= ∂x[I, J, K, L] * xμ * half * idenom²
            ∂sc[1, J, K, 1] += ∂y[I, J, K, L] * xμ * idenom
            ∂b[1, J, K, 1] += ∂y[I, J, K, L]
        end
    end

    return ∂x, ∂μ, ∂σ², ∂sc, ∂b
end

function __update_statistics(rμ, rσ², μ, σ², m1, m2)
    return __update_statistics(
        internal_operation_mode((rμ, rσ², μ, σ²)), rμ, rσ², μ, σ², m1, m2)
end

function __update_statistics(::GenericBroadcastOp, rμ, rσ², μ, σ², m1, m2)
    m3 = 1 - m1
    rμ2 = @. m3 * rμ + m1 * μ
    rσ²2 = @. m3 * rσ² + m2 * σ²
    return rμ2, rσ²2
end

function __update_statistics(opmode, rμ, rσ², μ, σ², m1, m2)
    m3 = 1 - m1
    rμ2 = similar(rμ, promote_type(eltype(rμ), eltype(μ), typeof(m3), typeof(m1)))
    rσ²2 = similar(rσ², promote_type(eltype(rσ²), eltype(σ²), typeof(m2), typeof(m3)))
    __update_statistics!(opmode, rμ2, rσ²2, rμ, rσ², μ, σ², m1, m2, 1 - m1)
    return rμ2, rσ²2
end
function __update_statistics!(::LoopedArrayOp, rμ2, rσ²2, rμ, rσ², μ, σ², m1, m2, m3)
    @simd ivdep for I in eachindex(rμ2, rσ²2)
        @inbounds rμ2[I] = m3 * rμ[I] + m1 * μ[I]
        @inbounds rσ²2[I] = m3 * rσ²[I] + m2 * σ²[I]
    end
end
function __update_statistics!(::GPUBroadcastOp, rμ2, rσ²2, rμ, rσ², μ, σ², m1, m2, m3)
    backend = KA.get_backend(rμ2)
    kernel! = __update_statistics_kernel!(backend)
    kernel!(rμ2, rσ²2, rμ, rσ², μ, σ², m1, m2, m3; ndrange=length(rμ2))
    KA.synchronize(backend)
end

@kernel function __update_statistics_kernel!(rμ2, rσ²2, @Const(rμ), @Const(rσ²), @Const(μ),
        @Const(σ²), @Const(m1), @Const(m2), @Const(m3))
    I = @index(Global)
    @inbounds rμ2[I] = m3 * rμ[I] + m1 * μ[I]
    @inbounds rσ²2[I] = m3 * rσ²[I] + m2 * σ²[I]
end

CRC.@non_differentiable __update_statistics(::Any...)

function _update_normalization_statistics(
        x::AbstractArray{T, N}, rμ::AbstractArray{<:Number, N},
        rσ²::AbstractArray{<:Number, N}, μ::AbstractArray{<:Number, N},
        σ²::AbstractArray{<:Number, N}, momentum::Real,
        r::Val{reduce_dims}) where {T, N, reduce_dims}
    if last(reduce_dims) != N
        μ = fast_mean(μ; dims=N)
        σ² = fast_mean(σ²; dims=N)
    end
    m = __value(T(__accum_size(x, r)))
    return __update_statistics(rμ, rσ², μ, σ², momentum, momentum * m / (m - one(m)))
end

CRC.@non_differentiable _update_normalization_statistics(::Any...)

__accum_size(x, ::Val{dims}) where {dims} = prod(Base.Fix1(size, x), dims)

function _get_batch_statistics(
        x::AbstractArray, ::Nothing, ::Nothing, ::Val{rdims}, ::Val, momentum) where {rdims}
    μ, σ² = fast_mean_var(x; dims=rdims, corrected=false)
    return (__aos_to_soa(μ), __aos_to_soa(σ²)), (nothing, nothing)
end

function _get_batch_statistics(::AbstractArray, rμ::AbstractArray, rσ²::AbstractArray,
        ::Val{rdims}, ::Val{false}, momentum) where {rdims}
    return (rμ, rσ²), (rμ, rσ²)
end

function _get_batch_statistics(x::AbstractArray, rμ::AbstractArray, rσ²::AbstractArray,
        r::Val{rdims}, ::Val{true}, momentum) where {rdims}
    μ, σ² = map(__aos_to_soa, fast_mean_var(x; dims=rdims, corrected=false))
    rμ, rσ² = _update_normalization_statistics(
        __value(x), __value(rμ), __value(rσ²), __value(μ), __value(σ²), momentum, r)
    return (μ, σ²), (rμ, rσ²)
end

# NOTE: marking it as stable makes everything type unstable in the backward pass
function _normalization(x::AbstractArray, running_mean::Optional{<:AbstractVector},
        running_var::Optional{<:AbstractVector}, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, reduce_dims::Val,
        training::Val, momentum, epsilon, act::F=identity) where {F}
    (μ, σ²), (rμ, rσ²) = _get_batch_statistics(
        x, _reshape_into_normalization_shape(running_mean, x),
        _reshape_into_normalization_shape(running_var, x), reduce_dims, training, momentum)
    return _affine_normalize(act, x, μ, σ², _reshape_into_normalization_shape(scale, x),
        _reshape_into_normalization_shape(bias, x), epsilon), _vec(rμ), _vec(rσ²)
end

_reshape_into_normalization_shape(::Nothing, y) = nothing
function _reshape_into_normalization_shape(x, y)
    return reshape(x, _get_norm_reshape_dims(size(y), length(x)))
end

@inbounds function _get_norm_reshape_dims(sx::NTuple{N, <:Int}, ly::Int) where {N}
    if ly == sx[N - 1]
        return ntuple(i -> i == N - 1 ? ly : 1, N)
    elseif N > 2 && ly == sx[N - 1] * sx[N - 2]
        return ntuple(i -> i == (N - 1) || i == (N - 2) ? sx[i] : 1, N)
    end
    throw(ArgumentError("Invalid Dimensions!"))
end

CRC.@non_differentiable _get_norm_reshape_dims(::Any...)
EnzymeRules.inactive_noinl(::typeof(_get_norm_reshape_dims), ::Any...) = nothing

# Generally you want to use `_normalization` but calling these functions lead to faster
# code.
function _groupnorm_impl(x::AbstractArray, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, reduce_dims::Val,
        epsilon, act::F=identity) where {F}
    (μ, σ²), _ = _get_batch_statistics(
        x, nothing, nothing, reduce_dims, Val(false), nothing)
    return _affine_normalize_gn(act, x, μ, σ², scale, bias, epsilon)
end

function _batchnorm_impl(x::AbstractArray, running_mean::Optional{<:AbstractVector},
        running_var::Optional{<:AbstractVector}, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, reduce_dims::Val,
        training::Val, momentum, epsilon, act::F=identity) where {F}
    (μ, σ²), (rμ, rσ²) = _get_batch_statistics(
        x, _reshape_into_normalization_shape(running_mean, x),
        _reshape_into_normalization_shape(running_var, x), reduce_dims, training, momentum)
    return _affine_normalize_bn(act, x, μ, σ², scale, bias, epsilon), _vec(rμ), _vec(rσ²)
end

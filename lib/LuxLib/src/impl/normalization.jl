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
end

@kernel function __update_statistics_kernel!(rμ2, rσ²2, @Const(rμ), @Const(rσ²), @Const(μ),
        @Const(σ²), @Const(m1), @Const(m2), @Const(m3))
    I = @index(Global)
    @inbounds rμ2[I] = m3 * rμ[I] + m1 * μ[I]
    @inbounds rσ²2[I] = m3 * rσ²[I] + m2 * σ²[I]
end

CRC.@non_differentiable __update_statistics(::Any...)
EnzymeRules.inactive_noinl(::typeof(__update_statistics), ::Any...) = nothing

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
# NOTE: The following leads to mixed activity not sure why
# EnzymeRules.inactive_noinl(::typeof(_update_normalization_statistics), ::Any...) = nothing

__accum_size(x, ::Val{dims}) where {dims} = prod(Base.Fix1(size, x), dims)

function _get_batch_statistics(
        x::AbstractArray, ::Nothing, ::Nothing, ::Val{rdims}, ::Val, momentum) where {rdims}
    μ = __aos_to_soa(fast_mean(x; dims=rdims))
    σ² = __aos_to_soa(fast_var(x; corrected=false, mean=μ, dims=rdims))
    return (μ, σ²), (nothing, nothing)
end

function _get_batch_statistics(::AbstractArray, rμ::AbstractArray, rσ²::AbstractArray,
        ::Val{rdims}, ::Val{false}, momentum) where {rdims}
    return (rμ, rσ²), (rμ, rσ²)
end

function _get_batch_statistics(x::AbstractArray, rμ::AbstractArray, rσ²::AbstractArray,
        r::Val{rdims}, ::Val{true}, momentum) where {rdims}
    μ = __aos_to_soa(fast_mean(x; dims=rdims))
    σ² = __aos_to_soa(fast_var(x; corrected=false, mean=μ, dims=rdims))
    rμ, rσ² = _update_normalization_statistics(
        __value(x), __value(rμ), __value(rσ²), __value(μ), __value(σ²), momentum, r)
    return (μ, σ²), (rμ, rσ²)
end

@stable default_mode="warn" function _normalization(
        x::AbstractArray, running_mean::Optional{<:AbstractVector},
        running_var::Optional{<:AbstractVector}, scale::Optional{<:AbstractVector},
        bias::Optional{<:AbstractVector}, reduce_dims::Val,
        training::Val, momentum, epsilon, act::F=identity) where {F}
    (μ, σ²), (rμ, rσ²) = _get_batch_statistics(
        x, _reshape_into_proper_shape(running_mean, x),
        _reshape_into_proper_shape(running_var, x), reduce_dims, training, momentum)
    return _affine_normalize(act, x, μ, σ², _reshape_into_proper_shape(scale, x),
        _reshape_into_proper_shape(bias, x), epsilon), _vec(rμ), _vec(rσ²)
end

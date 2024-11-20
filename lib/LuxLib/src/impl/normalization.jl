# In most cases this implementation should not be preferred. But this is nice to have
# because it works for arbitrary dimensions
function affine_normalize(act::F, x::AbstractArray, μ::Numeric, σ²::Numeric,
        ::Nothing, ::Nothing, ϵ::Real) where {F}
    γ′ = @. inv(sqrt(σ² + ϵ))
    β′ = @. -μ * γ′
    return @. act(x * γ′ + β′)
end

function affine_normalize(act::F, x::AbstractArray, μ::Numeric, σ²::Numeric,
        γ::AbstractArray, β::AbstractArray, ϵ::Real) where {F}
    γ′ = @. γ / sqrt(σ² + ϵ)
    β′ = @. β - μ * γ′
    return @. act(x * γ′ + β′)
end

# Deal with statistics
function update_running_statistics(rμ, rσ², μ, σ², m₁, m₂)
    return update_running_statistics(
        internal_operation_mode((rμ, rσ², μ, σ²)), rμ, rσ², μ, σ², m₁, m₂, 1 - m₁)
end

function update_running_statistics(::GenericBroadcastOp, rμ, rσ², μ, σ², m₁, m₂, m₃)
    rμₙ = @. m₃ * rμ + m₁ * μ
    rσ²ₙ = @. m₃ * rσ² + m₂ * σ²
    return rμₙ, rσ²ₙ
end

function update_running_statistics(opmode, rμ, rσ², μ, σ², m₁, m₂, m₃)
    rμₙ = similar(rμ, promote_type(eltype(rμ), eltype(μ), typeof(m₃), typeof(m₁)))
    rσ²ₙ = similar(rσ², promote_type(eltype(rσ²), eltype(σ²), typeof(m₂), typeof(m₃)))
    update_running_statistics!(rμₙ, rσ²ₙ, opmode, rμ, rσ², μ, σ², m₁, m₂, m₃)
    return rμₙ, rσ²ₙ
end

CRC.@non_differentiable update_running_statistics(::Any...)

function update_running_statistics!(rμₙ, rσ²ₙ, ::LoopedArrayOp, rμ, rσ², μ, σ², m₁, m₂, m₃)
    update_running_statistics_simd_loop!(
        rμₙ, rσ²ₙ, LoopedArrayOp(), rμ, rσ², μ, σ², m₁, m₂, m₃)
    return
end

function update_running_statistics_simd_loop!(
        rμₙ, rσ²ₙ, ::LoopedArrayOp, rμ, rσ², μ, σ², m₁, m₂, m₃)
    @simd ivdep for I in eachindex(rμₙ, rσ²ₙ)
        rμₙ[I] = m₃ * rμ[I] + m₁ * μ[I]
        rσ²ₙ[I] = m₃ * rσ²[I] + m₂ * σ²[I]
    end
end

function update_running_statistics!(rμₙ, rσ²ₙ, ::GPUBroadcastOp, rμ, rσ², μ, σ², m₁, m₂, m₃)
    backend = KA.get_backend(rμₙ)
    run_ka_kernel(
        update_running_statistics_kernel!, backend, nothing, size(rμₙ),
        rμₙ, rσ²ₙ, rμ, rσ², μ, σ², m₁, m₂, m₃)
    KA.synchronize(backend)
    return
end

@kernel cpu=false inbounds=true function update_running_statistics_kernel!(
        rμₙ, rσ²ₙ, @Const(rμ), @Const(rσ²), @Const(μ),
        @Const(σ²), @Const(m₁), @Const(m₂), @Const(m₃))
    I = @index(Global)
    rμₙ[I] = m₃ * rμ[I] + m₁ * μ[I]
    rσ²ₙ[I] = m₃ * rσ²[I] + m₂ * σ²[I]
end

function update_normalization_statistics(
        x::AbstractArray{T, N}, rμ::AbstractArray{rμT, N}, rσ²::AbstractArray{rσ²T, N},
        μ::AbstractArray{μT, N}, σ²::AbstractArray{σ²T, N},
        momentum::Real, reduce_dims) where {T, N, rμT, rσ²T, μT, σ²T}
    if last(reduce_dims) != N
        μ = mean(μ; dims=N)
        σ² = mean(σ²; dims=N)
    end
    m = remove_tracking(T(accum_size(x, reduce_dims)))
    return update_running_statistics(rμ, rσ², μ, σ², momentum, momentum * m / (m - one(m)))
end

accum_size(x, reduce_dims) = prod(Base.Fix1(size, x), unsafe_known(reduce_dims))

CRC.@non_differentiable update_normalization_statistics(::Any...)

function compute_batch_statistics(
        x::AbstractArray, ::Nothing, ::Nothing, reduce_dims, ::StaticBool, momentum)
    μ, σ² = mean_var(x; dims=unsafe_known(reduce_dims), corrected=false)
    return (aos_to_soa(μ), aos_to_soa(σ²)), (nothing, nothing)
end

function compute_batch_statistics(
        ::AbstractArray, rμ::AbstractArray, rσ²::AbstractArray, _, ::False, momentum)
    return (remove_tracking(rμ), remove_tracking(rσ²)), (rμ, rσ²)
end

function compute_batch_statistics(x::AbstractArray, rμ::AbstractArray,
        rσ²::AbstractArray, reduce_dims, ::True, momentum)
    μ, σ² = mean_var(x; dims=unsafe_known(reduce_dims), corrected=false)
    rμ, rσ² = update_normalization_statistics(
        remove_tracking(x), remove_tracking(rμ), remove_tracking(rσ²),
        remove_tracking(μ), remove_tracking(σ²), momentum, reduce_dims)
    return (aos_to_soa(μ), aos_to_soa(σ²)), (rμ, rσ²)
end

# Main Implementation
## The idea here is to be generic. This is useful for testing the more optimized
## implementations as well.
function normalization(
        x::AbstractArray, rμ::Optional{<:AbstractVector}, rσ²::Optional{<:AbstractVector},
        γ::Optional{<:AbstractVector}, β::Optional{<:AbstractVector}, reduce_dims,
        training::StaticBool, momentum, epsilon, act::F=identity) where {F}
    (μ, σ²), (rμ, rσ²) = compute_batch_statistics(
        x, reshape_norm_dims(x, rμ), reshape_norm_dims(x, rσ²),
        reduce_dims, training, momentum)
    γ, β = reshape_norm_dims(x, γ), reshape_norm_dims(x, β)
    return affine_normalize(act, x, μ, σ², γ, β, epsilon), rμ, rσ²
end

reshape_norm_dims(_, ::Nothing) = nothing
reshape_norm_dims(y, x) = reshape(x, get_norm_reshape_dims(size(y), length(x)))

@inbounds function get_norm_reshape_dims(sx::NTuple{N, <:Int}, ly::Int) where {N}
    if ly == sx[N - 1]
        return ntuple(i -> i == N - 1 ? ly : 1, N)
    elseif N > 2 && ly == sx[N - 1] * sx[N - 2]
        return ntuple(i -> i == (N - 1) || i == (N - 2) ? sx[i] : 1, N)
    end
    throw(ArgumentError("Invalid Dimensions!"))
end

CRC.@non_differentiable get_norm_reshape_dims(::Any...)
EnzymeRules.inactive(::typeof(get_norm_reshape_dims), ::Any...) = true

# Entry Points
## InstanceNorm
function instancenorm(x::AbstractArray{xT, N}, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, rμ::Optional{<:AbstractVector},
        rσ²::Optional{<:AbstractVector}, training::StaticBool,
        act::F, momentum, epsilon) where {xT, N, F}
    y, rμₙ, rσ²ₙ = normalization(
        x, rμ, rσ², γ, β, instancenorm_reduce_dims(x), training, momentum, epsilon, act)
    return y, safe_vec(rμₙ), safe_vec(rσ²ₙ)
end

instancenorm_reduce_dims(::AbstractArray{T, N}) where {T, N} = ntuple(static, N - 2)

CRC.@non_differentiable instancenorm_reduce_dims(::Any...)

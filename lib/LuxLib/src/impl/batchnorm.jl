function batchnorm_cudnn end   # Defined in LuxLibcuDNNExt
function ∇batchnorm_cudnn end  # Defined in LuxLibcuDNNExt

function batchnorm_reduce_dims(::AbstractArray{T, N}) where {T, N}
    return (ntuple(static, N - 2)..., static(N))
end

CRC.@non_differentiable batchnorm_reduce_dims(::Any...)

function get_batchnorm_statistics(::AbstractArray, rμ::Optional{<:AbstractVector},
        rσ²::Optional{<:AbstractVector}, ::True)
    return Utils.copy_drop_gradients(rμ), Utils.copy_drop_gradients(rσ²)
end

function get_batchnorm_statistics(x::AbstractArray, ::Nothing, ::Nothing, ::False)
    μ, σ² = mean_var(x; dims=Utils.known(batchnorm_reduce_dims(x)), corrected=false)
    return Utils.vec(μ), Utils.vec(σ²)
end

function get_batchnorm_statistics(
        ::AbstractArray, rμ::AbstractVector, rσ²::AbstractVector, ::False)
    return rμ, rσ²
end

CRC.@non_differentiable get_batchnorm_statistics(::Any...)

function batchnorm(x::AbstractArray{<:Number, N}, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, rμ::Optional{<:AbstractVector},
        rσ²::Optional{<:AbstractVector}, training::StaticBool,
        act::F, momentum::Real, ϵ::Real) where {F, N}
    (μ, σ²), (rμ, rσ²) = compute_batch_statistics(
        x, reshape_norm_dims(x, rμ), reshape_norm_dims(x, rσ²),
        batchnorm_reduce_dims(x), training, momentum)
    return (batchnorm_affine_normalize(act, x, μ, σ², γ, β, ϵ),
        get_utils(:vec)(rμ), get_utils(:vec)(rσ²))
end

function batchnorm_affine_normalize(
        act::F, x::AbstractArray{<:Number, N}, μ::AbstractArray{<:Number, N},
        σ²::AbstractArray{<:Number, N}, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, ϵ::Real) where {N, F}
    return batchnorm_affine_normalize(
        internal_operation_mode((x, μ, σ², γ, β)), act, x, μ, σ², γ, β, ϵ)
end

function batchnorm_affine_normalize(
        ::GenericBroadcastOp, act::F, x::AbstractArray{<:Number, N},
        μ::AbstractArray{<:Number, N}, σ²::AbstractArray{<:Number, N},
        γ::Optional{<:AbstractVector}, β::Optional{<:AbstractVector}, ϵ::Real) where {F, N}
    return affine_normalize(
        act, x, μ, σ², reshape_norm_dims(x, γ), reshape_norm_dims(x, β), ϵ)
end

function batchnorm_affine_normalize(
        opmode::AbstractInternalArrayOpMode, act::F, x::AbstractArray{<:Number, N},
        μ::AbstractArray{<:Number, N}, σ²::AbstractArray{<:Number, N},
        γ::Optional{<:AbstractVector}, β::Optional{<:AbstractVector}, ϵ::Real) where {F, N}
    x′ = reshape(x, :, size(x, N - 1), size(x, N))
    return reshape(
        batchnorm_affine_normalize_internal(opmode, act, x′, vec(μ), vec(σ²), γ, β, ϵ),
        size(x))
end

@stable default_mode="disable" function batchnorm_affine_normalize_internal(
        opmode::AbstractInternalArrayOpMode, act::F, x::AbstractArray{<:Number, 3},
        μ::AbstractVector, σ²::AbstractVector, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, ϵ::Real) where {F}
    y = similar(x,
        promote_type(Utils.eltype(x), Utils.eltype(μ), Utils.eltype(σ²),
            Utils.eltype(γ), Utils.eltype(β)))
    batchnorm_affine_normalize_internal!(y, opmode, act, x, μ, σ², γ, β, ϵ)
    return y
end

function batchnorm_affine_normalize_internal!(
        y::AbstractArray{<:Number, 3}, opmode::LoopedArrayOp, act::F,
        x::AbstractArray{<:Number, 3}, μ::AbstractVector, σ²::AbstractVector,
        γ::Optional{<:AbstractVector}, β::Optional{<:AbstractVector},
        ϵ::Real, γ′::Optional{<:AbstractVector}=nothing) where {F}
    N = size(y, 2)
    γ′ = γ′ === nothing ?
         similar(x, promote_type(Utils.eltype(γ), Utils.eltype(σ²), Utils.eltype(ϵ)), N) :
         γ′
    β′ = similar(x, promote_type(Utils.eltype(β), Utils.eltype(σ²), Utils.eltype(ϵ)), N)

    compute_batchnorm_scale_bias_loopvec!(γ′, β′, γ, β, μ, σ², ϵ)
    apply_batchnorm_scale_bias!(y, γ′, β′, x)
    activation!(y, opmode, act, y)
    return
end

function compute_batchnorm_scale_bias_loopvec!(γ′, β′, ::Nothing, ::Nothing, μ, σ², ϵ)
    if LV.check_args(γ′, β′, μ, σ²)
        @tturbo for J in indices((γ′, β′, μ, σ²))
            γ′[J] = inv(sqrt(σ²[J] + ϵ))
            β′[J] = -μ[J] * γ′[J]
        end
    else
        @batch for J in indices((γ′, β′, μ, σ²))
            @inbounds γ′[J] = inv(sqrt(σ²[J] + ϵ))
            @inbounds β′[J] = -μ[J] * γ′[J]
        end
    end
end

function compute_batchnorm_scale_bias_loopvec!(γ′, β′, γ, β, μ, σ², ϵ)
    if LV.check_args(γ′, β′, γ, β, μ, σ²)
        @tturbo for J in indices((γ′, β′, γ, β, μ, σ²))
            γ′[J] = γ[J] / sqrt(σ²[J] + ϵ)
            β′[J] = β[J] - μ[J] * γ′[J]
        end
    else
        @batch for J in indices((γ′, β′, γ, β, μ, σ²))
            @inbounds γ′[J] = γ[J] / sqrt(σ²[J] + ϵ)
            @inbounds β′[J] = β[J] - μ[J] * γ′[J]
        end
    end
end

function compute_batchnorm_scale_bias_simd_loop!(γ′, β′, ::Nothing, ::Nothing, μ, σ², ϵ)
    @simd ivdep for J in indices((γ′, β′, μ, σ²))
        @inbounds γ′[J] = inv(sqrt(σ²[J] + ϵ))
        @inbounds β′[J] = -μ[J] * γ′[J]
    end
end

function compute_batchnorm_scale_bias_simd_loop!(γ′, β′, γ, β, μ, σ², ϵ)
    @simd ivdep for J in indices((γ′, β′, γ, β, μ, σ²))
        @inbounds γ′[J] = γ[J] / sqrt(σ²[J] + ϵ)
        @inbounds β′[J] = β[J] - μ[J] * γ′[J]
    end
end

Utils.@enzyme_reverse_alternative compute_batchnorm_scale_bias_loopvec! compute_batchnorm_scale_bias_simd_loop!

function apply_batchnorm_scale_bias!(y::AbstractArray{<:Number, 3}, γ′::AbstractVector,
        β′::AbstractVector, x::AbstractArray{<:Number, 3})
    if LV.check_args(y, γ′, β′, x)
        @tturbo for K in indices((x, y), 3),
            J in indices((x, y, γ′, β′), (2, 2, 1, 1)),
            I in indices((x, y), 1)

            y[I, J, K] = x[I, J, K] * γ′[J] + β′[J]
        end
    else
        @batch for K in indices((x, y), 3), J in indices((x, y, γ′, β′), (2, 2, 1, 1))
            @simd ivdep for I in indices((x, y), 1)
                @inbounds y[I, J, K] = x[I, J, K] * γ′[J] + β′[J]
            end
        end
    end
end

function apply_batchnorm_scale_bias_simd_loop!(
        y::AbstractArray{<:Number, 3}, γ′::AbstractVector,
        β′::AbstractVector, x::AbstractArray{<:Number, 3})
    for K in indices((x, y), 3), J in indices((x, y, γ′, β′), (2, 2, 1, 1))
        @simd ivdep for I in indices((x, y), 1)
            @inbounds y[I, J, K] = x[I, J, K] * γ′[J] + β′[J]
        end
    end
end

Utils.@enzyme_reverse_alternative apply_batchnorm_scale_bias! apply_batchnorm_scale_bias_simd_loop!

function batchnorm_affine_normalize_internal!(
        y::AbstractArray{<:Number, 3}, ::GPUBroadcastOp, act::F,
        x::AbstractArray{<:Number, 3}, μ::AbstractVector, σ²::AbstractVector,
        γ::Optional{<:AbstractVector}, β::Optional{<:AbstractVector},
        ϵ::Real, γ′::Optional{<:AbstractVector}=nothing) where {F}
    backend = KA.get_backend(y)
    if γ′ === nothing
        kernel! = batchnorm_affine_normalize_internal_kernel!(backend)
        kernel!(y, act, x, μ, σ², γ, β, ϵ; ndrange=size(y))
    else
        kernel! = batchnorm_affine_normalize_internal_kernel_cached!(backend)
        kernel!(y, γ′, act, x, μ, σ², γ, β, ϵ; ndrange=size(y))
    end
    KA.synchronize(backend)
end

@kernel function batchnorm_affine_normalize_internal_kernel!(
        y::AbstractArray{<:Number, 3}, @Const(f), @Const(x),
        @Const(μ), @Const(σ²), @Const(γ), @Const(β), @Const(ϵ))
    (i, j, k) = @index(Global, NTuple)
    if γ !== nothing
        @inbounds γ′ = γ[j] / sqrt(σ²[j] + ϵ)
        @inbounds β′ = muladd(-μ[j], γ′, β[j])
    else
        @inbounds γ′ = inv(sqrt(σ²[j] + ϵ))
        @inbounds β′ = -μ[j] * γ′
    end
    @inbounds y[i, j, k] = f(muladd(x[i, j, k], γ′, β′))
end

@kernel function batchnorm_affine_normalize_internal_kernel_cached!(
        y::AbstractArray{<:Number, 3}, γ′::AbstractVector{<:Number}, @Const(f),
        @Const(x), @Const(μ), @Const(σ²), @Const(γ), @Const(β), @Const(ϵ))
    (i, j, k) = @index(Global, NTuple)
    if γ !== nothing
        @inbounds γ′[j] = γ[j] / sqrt(σ²[j] + ϵ)
        @inbounds β′ = muladd(-μ[j], γ′[j], β[j])
    else
        @inbounds γ′[j] = inv(sqrt(σ²[j] + ϵ))
        @inbounds β′ = -μ[j] * γ′[j]
    end
    @inbounds y[i, j, k] = f(muladd(x[i, j, k], γ′[j], β′))
end

function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(batchnorm_affine_normalize_internal),
        opmode::AbstractInternalArrayOpMode, act::F, x::AbstractArray{T, N},
        μ::AbstractVector, σ²::AbstractVector, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, ϵ::Real) where {F, T, N}
    y = similar(x,
        promote_type(Utils.eltype(x), Utils.eltype(μ), Utils.eltype(σ²),
            Utils.eltype(γ), Utils.eltype(β)))
    γ′ = similar(
        x, promote_type(Utils.eltype(γ), Utils.eltype(σ²), Utils.eltype(ϵ)), size(x, N - 1))

    batchnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ, γ′)
    z, ∇activation = CRC.rrule_via_ad(
        cfg, activation!!, opmode, Traits.is_mutable_array(y), act, y)

    𝒫x, 𝒫μ, 𝒫σ² = CRC.ProjectTo(x), CRC.ProjectTo(μ), CRC.ProjectTo(σ²)
    𝒫γ = γ === nothing ? identity : CRC.ProjectTo(γ)
    𝒫β = β === nothing ? identity : CRC.ProjectTo(β)

    ∇batchnorm_affine_normalize_internal = @closure Δ -> begin
        ∂y = last(∇activation(Δ))
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇batchnorm_affine_normalize(opmode, ∂y, x, μ, σ², γ, β, ϵ, γ′)
        return ∂∅, ∂∅, ∂∅, 𝒫x(∂x), 𝒫μ(∂μ), 𝒫σ²(∂σ²), 𝒫γ(∂γ), 𝒫β(∂β), ∂∅
    end

    return z, ∇batchnorm_affine_normalize_internal
end

function ∇batchnorm_affine_normalize(
        opmode::AbstractInternalArrayOpMode, ∂y::AbstractArray{<:Number, 3},
        x::AbstractArray{<:Number, 3}, μ::AbstractVector,
        σ²::AbstractVector, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, ϵ::Real, γ′::AbstractVector)
    ∂x, ∂σ² = similar(x), similar(σ², size(x))
    ∂γ = γ === nothing ? nothing : similar(γ, size(x))

    ∇batchnorm_affine_normalize!(∂x, ∂σ², ∂γ, opmode, ∂y, x, μ, σ², γ, ϵ, γ′)

    ∂μ = dropdims(sum(-, ∂x; dims=(1, 3)); dims=(1, 3))
    ∂σ² = dropdims(sum(∂σ²; dims=(1, 3)); dims=(1, 3))
    ∂γ = γ === nothing ? ∂∅ : dropdims(sum(∂γ; dims=(1, 3)); dims=(1, 3))
    ∂β = β === nothing ? ∂∅ : dropdims(sum(∂y; dims=(1, 3)); dims=(1, 3))

    return ∂x, ∂μ, ∂σ², ∂γ, ∂β
end

function ∇batchnorm_affine_normalize!(
        ∂x::AbstractArray{<:Number, 3}, ∂σ²::AbstractArray{<:Number, 3}, ::Nothing,
        ::LoopedArrayOp, ∂y::AbstractArray{<:Number, 3}, x::AbstractArray{<:Number, 3},
        μ::AbstractVector, σ²::AbstractVector, ::Nothing, ϵ::Real, γ′::AbstractVector)
    half = eltype(∂σ²)(0.5)

    if LV.check_args(∂x, ∂σ², ∂y, x, μ, σ²)
        @tturbo for K in indices(∂y, 3), J in indices(∂y, 2)
            idenom = γ′[J]
            idenom² = idenom^2

            for I in indices(∂y, 1)
                xμ = x[I, J, K] - μ[J]

                ∂x[I, J, K] = ∂y[I, J, K] * idenom
                ∂σ²[I, J, K] = -∂x[I, J, K] * xμ * half * idenom²
            end
        end
    else
        @inbounds @batch for K in indices(∂y, 3), J in indices(∂y, 2)
            idenom = γ′[J]
            idenom² = idenom^2

            @simd for I in indices(∂y, 1)
                xμ = x[I, J, K] - μ[J]

                ∂x[I, J, K] = ∂y[I, J, K] * idenom
                ∂σ²[I, J, K] = -∂x[I, J, K] * xμ * half * idenom²
            end
        end
    end
end

function ∇batchnorm_affine_normalize!(
        ∂x::AbstractArray{<:Number, 3}, ∂σ²::AbstractArray{<:Number, 3},
        ∂γ::AbstractArray{<:Number, 3}, ::LoopedArrayOp, ∂y::AbstractArray{<:Number, 3},
        x::AbstractArray{<:Number, 3}, μ::AbstractVector,
        σ²::AbstractVector, γ::AbstractVector, ϵ::Real, γ′::AbstractVector)
    half = eltype(∂σ²)(0.5)

    if LV.check_args(∂x, ∂σ², ∂γ, ∂y, x, μ, σ², γ)
        @tturbo for K in indices(∂y, 3), J in indices(∂y, 2)
            idenom = inv(sqrt(σ²[J] + ϵ))
            idenom² = idenom^2

            for I in indices(∂y, 1)
                xμ = x[I, J, K] - μ[J]

                ∂x[I, J, K] = ∂y[I, J, K] * γ′[J]
                ∂σ²[I, J, K] = -∂x[I, J, K] * xμ * half * idenom²
                ∂γ[I, J, K] = ∂y[I, J, K] * xμ * idenom
            end
        end
    else
        @inbounds @batch for K in indices(∂y, 3), J in indices(∂y, 2)
            idenom = inv(sqrt(σ²[J] + ϵ))
            idenom² = idenom^2

            @simd for I in indices(∂y, 1)
                xμ = x[I, J, K] - μ[J]

                ∂x[I, J, K] = ∂y[I, J, K] * γ′[J]
                ∂σ²[I, J, K] = -∂x[I, J, K] * xμ * half * idenom²
                ∂γ[I, J, K] = ∂y[I, J, K] * xμ * idenom
            end
        end
    end
end

function ∇batchnorm_affine_normalize!(
        ∂x::AbstractArray{<:Number, 3}, ∂σ²::AbstractArray{<:Number, 3},
        ∂γ::Optional{<:AbstractArray{<:Number, 3}}, ::GPUBroadcastOp,
        ∂y::AbstractArray{<:Number, 3}, x::AbstractArray{<:Number, 3}, μ::AbstractVector,
        σ²::AbstractVector, γ::Optional{<:AbstractVector}, ϵ::Real, γ′::AbstractVector)
    backend = KA.get_backend(∂x)
    kernel! = ∇batchnorm_affine_normalize_kernel!(backend)
    kernel!(∂x, ∂σ², ∂γ, ∂y, x, μ, σ², γ, ϵ, γ′; ndrange=size(∂x))
    KA.synchronize(backend)
end

@kernel function ∇batchnorm_affine_normalize_kernel!(
        ∂x, ∂σ², ∂γ, @Const(∂y), @Const(x), @Const(μ),
        @Const(σ²), @Const(γ), @Const(ϵ), @Const(γ′))
    (i, j, k) = @index(Global, NTuple)
    if γ !== nothing
        @inbounds idenom = inv(sqrt(σ²[j] + ϵ))
    else
        @inbounds idenom = γ′[j]
    end
    idenom² = idenom^2

    @inbounds xμ = x[i, j, k] - μ[j]

    @inbounds ∂x[i, j, k] = ∂y[i, j, k] * γ′[j]
    @inbounds ∂σ²[i, j, k] = -∂x[i, j, k] * xμ * idenom² / 2

    if γ !== nothing
        @inbounds ∂γ[i, j, k] = ∂y[i, j, k] * xμ * idenom
    end
end

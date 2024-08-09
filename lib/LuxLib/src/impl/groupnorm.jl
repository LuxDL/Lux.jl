groupnorm_reduce_dims(::AbstractArray{T, N}) where {T, N} = ntuple(static, N - 1)

function groupnorm(x::AbstractArray{<:Number, N}, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, groups::Int, act::F, ϵ::Real) where {F, N}
    x′ = reshape(x, size(x)[1:(N - 2)]..., size(x, N - 1) ÷ groups, groups, size(x, N))
    (μ, σ²), _ = compute_batch_statistics(
        x′, nothing, nothing, groupnorm_reduce_dims(x), False(), nothing)
    return reshape(groupnorm_affine_normalize(act, x′, μ, σ², γ, β, ϵ), size(x))
end

function groupnorm_affine_normalize(
        act::F, x::AbstractArray{<:Number, N}, μ::AbstractArray{<:Number, N},
        σ²::AbstractArray{<:Number, N}, γ::Optional{<:AbstractVector},
        β::Optional{<:AbstractVector}, ϵ::Real) where {F, N}
    return groupnorm_affine_normalize(
        internal_operation_mode((x, μ, σ², γ, β)), act, x, μ, σ², γ, β, ϵ)
end

function groupnorm_affine_normalize(
        ::GenericBroadcastOp, act::F, x::AbstractArray{<:Number, N},
        μ::AbstractArray{<:Number, N}, σ²::AbstractArray{<:Number, N},
        γ::Optional{<:AbstractVector}, β::Optional{<:AbstractVector}, ϵ::Real) where {F, N}
    return affine_normalize(
        act, x, μ, σ², reshape_norm_dims(x, γ), reshape_norm_dims(x, β), ϵ)
end

function groupnorm_affine_normalize(
        opmode::AbstractInternalArrayOpMode, act::F, x::AbstractArray{<:Number, N},
        μ::AbstractArray{<:Number, N}, σ²::AbstractArray{<:Number, N},
        γ::Optional{<:AbstractVector}, β::Optional{<:AbstractVector}, ϵ::Real) where {F, N}
    x′ = reshape(x, :, size(x, N - 2), size(x, N - 1), size(x, N))
    μ′ = reshape(μ, 1, 1, size(x, N - 1), size(x, N))
    σ²′ = reshape(σ², 1, 1, size(x, N - 1), size(x, N))
    γ′ = Utils.reshape(γ, 1, size(x, N - 2), size(x, N - 1), 1)
    β′ = Utils.reshape(β, 1, size(x, N - 2), size(x, N - 1), 1)

    return reshape(
        groupnorm_affine_normalize_internal(opmode, act, x′, μ′, σ²′, γ′, β′, ϵ), size(x))
end

function groupnorm_affine_normalize_internal(opmode::AbstractInternalArrayOpMode, act::F,
        x::AbstractArray{<:Number, 4}, μ::AbstractArray{<:Number, 4},
        σ²::AbstractArray{<:Number, 4}, γ::Optional{<:AbstractArray{<:Number, 4}},
        β::Optional{<:AbstractArray{<:Number, 4}}, ϵ::Real) where {F}
    y = similar(x,
        promote_type(Utils.eltype(x), Utils.eltype(μ), Utils.eltype(σ²),
            Utils.eltype(γ), Utils.eltype(β)))
    groupnorm_affine_normalize_internal!(y, opmode, act, x, μ, σ², γ, β, ϵ)
    return y
end

function groupnorm_affine_normalize_internal!(
        y::AbstractArray{<:Number, 4}, opmode::LoopedArrayOp, act::F,
        x::AbstractArray{<:Number, 4}, μ::AbstractArray{<:Number, 4},
        σ²::AbstractArray{<:Number, 4}, γ::Optional{<:AbstractArray{<:Number, 4}},
        β::Optional{<:AbstractArray{<:Number, 4}}, ϵ::Real) where {F}
    affine_normalize_loopvec!(y, x, μ, σ², γ, β, ϵ)
    activation!(y, opmode, act, y)
    return
end

function affine_normalize_loopvec!(
        y::AbstractArray{<:Number, 4}, x::AbstractArray{<:Number, 4},
        μ::AbstractArray{<:Number, 4},
        σ²::AbstractArray{<:Number, 4}, ::Nothing, ::Nothing, ϵ::Real)
    if LV.check_args(y, x, μ, σ², ϵ)
        @tturbo for L in indices(y, 4), K in indices(y, 3)
            γ′ = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            β′ = -μ[1, 1, K, L] * γ′
            for J in indices(y, 2), I in indices(y, 1)
                y[I, J, K, L] = muladd(x[I, J, K, L], γ′, β′)
            end
        end
    else
        @inbounds @batch for L in indices(y, 4), K in indices(y, 3)
            γ′ = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            β′ = -μ[1, 1, K, L] * γ′
            for J in indices(y, 2)
                @simd ivdep for I in indices(y, 1)
                    y[I, J, K, L] = muladd(x[I, J, K, L], γ′, β′)
                end
            end
        end
    end
end

function affine_normalize_loopvec!(
        y::AbstractArray{<:Number, 4}, x::AbstractArray{<:Number, 4},
        μ::AbstractArray{<:Number, 4}, σ²::AbstractArray{<:Number, 4},
        γ::AbstractArray{<:Number, 4}, β::AbstractArray{<:Number, 4}, ϵ::Real)
    if LV.check_args(y, x, μ, σ², γ, β, ϵ)
        @tturbo for L in indices(y, 4), K in indices(y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            for J in indices(y, 2)
                γ′ = γ[1, J, K, 1] * idenom
                β′ = muladd(-μ[1, 1, K, L], γ′, β[1, J, K, 1])
                for I in indices(y, 1)
                    y[I, J, K, L] = muladd(x[I, J, K, L], γ′, β′)
                end
            end
        end
    else
        @inbounds @batch for L in indices(y, 4), K in indices(y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            for J in indices(y, 2)
                γ′ = γ[1, J, K, 1] * idenom
                β′ = muladd(-μ[1, 1, K, L], γ′, β[1, J, K, 1])
                @simd ivdep for I in indices(y, 1)
                    y[I, J, K, L] = muladd(x[I, J, K, L], γ′, β′)
                end
            end
        end
    end
end

function affine_normalize_simd_loop!(
        y::AbstractArray{<:Number, 4}, x::AbstractArray{<:Number, 4},
        μ::AbstractArray{<:Number, 4},
        σ²::AbstractArray{<:Number, 4}, ::Nothing, ::Nothing, ϵ::Real)
    @inbounds for L in indices(y, 4), K in indices(y, 3)
        γ′ = inv(sqrt(σ²[1, 1, K, L] + ϵ))
        β′ = -μ[1, 1, K, L] * γ′
        for J in indices(y, 2)
            @simd ivdep for I in indices(y, 1)
                y[I, J, K, L] = muladd(x[I, J, K, L], γ′, β′)
            end
        end
    end
end

function affine_normalize_simd_loop!(
        y::AbstractArray{<:Number, 4}, x::AbstractArray{<:Number, 4},
        μ::AbstractArray{<:Number, 4}, σ²::AbstractArray{<:Number, 4},
        γ::AbstractArray{<:Number, 4}, β::AbstractArray{<:Number, 4}, ϵ::Real)
    @inbounds for L in indices(y, 4), K in indices(y, 3)
        idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
        for J in indices(y, 2)
            γ′ = γ[1, J, K, 1] * idenom
            β′ = muladd(-μ[1, 1, K, L], γ′, β[1, J, K, 1])
            @simd ivdep for I in indices(y, 1)
                y[I, J, K, L] = muladd(x[I, J, K, L], γ′, β′)
            end
        end
    end
end

Utils.@enzyme_reverse_alternative affine_normalize_loopvec! affine_normalize_simd_loop!

function groupnorm_affine_normalize_internal!(
        y::AbstractArray{<:Number, 4}, ::GPUBroadcastOp, act::F,
        x::AbstractArray{<:Number, 4}, μ::AbstractArray{<:Number, 4},
        σ²::AbstractArray{<:Number, 4}, γ::Optional{<:AbstractArray{<:Number, 4}},
        β::Optional{<:AbstractArray{<:Number, 4}}, ϵ::Real) where {F}
    backend = KA.get_backend(y)
    kernel! = groupnorm_affine_normalize_kernel!(backend)
    kernel!(y, act, x, μ, σ², γ, β, ϵ; ndrange=size(y))
    KA.synchronize(backend)
end

@kernel function groupnorm_affine_normalize_kernel!(
        y::AbstractArray{<:Number, 4}, @Const(f), @Const(x),
        @Const(μ), @Const(σ²), @Const(γ), @Const(β), @Const(ϵ))
    (i, j, k, l) = @index(Global, NTuple)
    if γ !== nothing
        @inbounds γ′ = γ[1, j, k, 1] / sqrt(σ²[1, 1, k, l] + ϵ)
        @inbounds β′ = muladd(-μ[1, 1, k, l], γ′, β[1, j, k, 1])
    else
        @inbounds γ′ = inv(sqrt(σ²[1, 1, k, l] + ϵ))
        @inbounds β′ = -μ[1, 1, k, l] * γ′
    end
    @inbounds y[i, j, k, l] = f(muladd(x[i, j, k, l], γ′, β′))
end

function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(groupnorm_affine_normalize_internal),
        opmode::AbstractInternalArrayOpMode, f::F,
        x::AbstractArray{T, 4}, μ::AbstractArray{<:Number, 4},
        σ²::AbstractArray{<:Number, 4}, γ::Optional{<:AbstractArray{<:Number, 4}},
        β::Optional{<:AbstractArray{<:Number, 4}}, ϵ::Real) where {F, T}
    y = similar(x,
        promote_type(Utils.eltype(x), Utils.eltype(μ), Utils.eltype(σ²),
            Utils.eltype(γ), Utils.eltype(β)))
    groupnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ)
    z, ∇activation = CRC.rrule_via_ad(cfg, activation!!, f, y)

    𝒫x, 𝒫μ, 𝒫σ² = CRC.ProjectTo(x), CRC.ProjectTo(μ), CRC.ProjectTo(σ²)
    𝒫γ = γ === nothing ? identity : CRC.ProjectTo(γ)
    𝒫β = β === nothing ? identity : CRC.ProjectTo(β)

    ∇groupnorm_affine_normalize_internal = @closure Δ -> begin
        ∂y = last(∇activation(Δ))
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇groupnorm_affine_normalize(opmode, ∂y, x, μ, σ², γ, β, ϵ)
        return ∂∅, ∂∅, ∂∅, 𝒫x(∂x), 𝒫μ(∂μ), 𝒫σ²(∂σ²), 𝒫γ(∂γ), 𝒫β(∂β), ∂∅
    end

    return z, ∇groupnorm_affine_normalize_internal
end

function ∇groupnorm_affine_normalize(
        opmode::AbstractInternalArrayOpMode, ∂y::AbstractArray{<:Number, 4},
        x::AbstractArray{<:Number, 4}, μ::AbstractArray{<:Number, 4},
        σ²::AbstractArray{<:Number, 4}, γ::Optional{<:AbstractArray{<:Number, 4}},
        β::Optional{<:AbstractArray{<:Number, 4}}, ϵ::Real)
    ∂x, ∂σ² = similar(x), similar(σ², size(x))
    ∂γ = γ === nothing ? nothing : similar(γ, size(x))

    ∇groupnorm_affine_normalize!(∂x, ∂σ², ∂γ, opmode, ∂y, x, μ, σ², γ, ϵ)

    ∂μ = sum(-, ∂x; dims=(1, 2))
    ∂σ² = sum(∂σ²; dims=(1, 2))
    ∂γ = γ === nothing ? ∂∅ : sum(∂γ; dims=(1, 4))
    ∂β = β === nothing ? ∂∅ : sum(∂y; dims=(1, 4))

    return ∂x, ∂μ, ∂σ², ∂γ, ∂β
end

function ∇groupnorm_affine_normalize!(
        ∂x::AbstractArray{<:Number, 4}, ∂σ²::AbstractArray{<:Number, 4}, ::Nothing,
        ::LoopedArrayOp, ∂y::AbstractArray{<:Number, 4}, x::AbstractArray{<:Number, 4},
        μ::AbstractArray{<:Number, 4}, σ²::AbstractArray{<:Number, 4}, ::Nothing, ϵ::Real)
    half = eltype(∂σ²)(0.5)

    if LV.check_args(∂x, ∂σ², ∂y, x, μ, σ², ϵ)
        @tturbo for L in indices(∂y, 4), K in indices(∂y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            idenom² = idenom^2

            for J in indices(∂y, 2), I in indices(∂y, 1)
                xμ = x[I, J, K, L] - μ[1, 1, K, L]

                ∂x[I, J, K, L] = ∂y[I, J, K, L] * idenom
                ∂σ²[I, J, K, L] = -∂x[I, J, K, L] * xμ * half * idenom²
            end
        end
    else
        @inbounds @batch for L in indices(∂y, 4), K in indices(∂y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            idenom² = idenom^2

            for J in indices(∂y, 2)
                @simd for I in indices(∂y, 1)
                    xμ = x[I, J, K, L] - μ[1, 1, K, L]

                    ∂x[I, J, K, L] = ∂y[I, J, K, L] * idenom
                    ∂σ²[I, J, K, L] = -∂x[I, J, K, L] * xμ * half * idenom²
                end
            end
        end
    end
end

function ∇groupnorm_affine_normalize!(
        ∂x::AbstractArray{<:Number, 4}, ∂σ²::AbstractArray{<:Number, 4},
        ∂γ::AbstractArray{<:Number, 4}, ::LoopedArrayOp, ∂y::AbstractArray{<:Number, 4},
        x::AbstractArray{<:Number, 4}, μ::AbstractArray{<:Number, 4},
        σ²::AbstractArray{<:Number, 4}, γ::AbstractArray{<:Number, 4}, ϵ::Real)
    half = eltype(∂σ²)(0.5)

    if LV.check_args(∂x, ∂σ², ∂γ, ∂y, x, μ, σ², γ, ϵ)
        @tturbo for L in indices(∂y, 4), K in indices(∂y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            idenom² = idenom^2

            for J in indices(∂y, 2)
                γ′ = γ[1, J, K, 1] * idenom
                for I in indices(∂y, 1)
                    xμ = x[I, J, K, L] - μ[1, 1, K, L]

                    ∂x[I, J, K, L] = ∂y[I, J, K, L] * γ′
                    ∂σ²[I, J, K, L] = -∂x[I, J, K, L] * xμ * half * idenom²
                    ∂γ[I, J, K, 1] = ∂y[I, J, K, L] * xμ * idenom
                end
            end
        end
    else
        @inbounds @batch for L in indices(∂y, 4), K in indices(∂y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            idenom² = idenom^2

            for J in indices(∂y, 2)
                γ′ = γ[1, J, K, 1] * idenom
                @simd for I in indices(∂y, 1)
                    xμ = x[I, J, K, L] - μ[1, 1, K, L]

                    ∂x[I, J, K, L] = ∂y[I, J, K, L] * γ′
                    ∂σ²[I, J, K, L] = -∂x[I, J, K, L] * xμ * half * idenom²
                    ∂γ[I, J, K, 1] = ∂y[I, J, K, L] * xμ * idenom
                end
            end
        end
    end
end

function ∇groupnorm_affine_normalize!(
        ∂x::AbstractArray{<:Number, 4}, ∂σ²::AbstractArray{<:Number, 4},
        ∂γ::Optional{<:AbstractArray{<:Number, 4}}, ::GPUBroadcastOp,
        ∂y::AbstractArray{<:Number, 4}, x::AbstractArray{<:Number, 4},
        μ::AbstractArray{<:Number, 4}, σ²::AbstractArray{<:Number, 4},
        γ::Optional{<:AbstractArray{<:Number, 4}}, ϵ::Real)
    backend = KA.get_backend(∂x)
    kernel! = ∇groupnorm_affine_normalize_kernel!(backend)
    kernel!(∂x, ∂σ², ∂γ, ∂y, x, μ, σ², γ, ϵ; ndrange=size(∂x))
    KA.synchronize(backend)
end

@kernel function ∇groupnorm_affine_normalize_kernel!(
        ∂x, ∂σ², ∂γ, @Const(∂y), @Const(x), @Const(μ), @Const(σ²), @Const(γ), @Const(ϵ))
    (i, j, k, l) = @index(Global, NTuple)
    @inbounds idenom = sqrt(σ²[1, 1, k, l] + ϵ)
    @inbounds idenom² = idenom^2

    if γ !== nothing
        @inbounds γ′ = γ[1, j, k, 1] / idenom
    else
        @inbounds γ′ = inv(idenom)
    end

    @inbounds xμ = x[i, j, k, l] - μ[1, 1, k, l]

    @inbounds ∂x[i, j, k, l] = ∂y[i, j, k, l] * γ′
    @inbounds ∂σ²[i, j, k, l] = -∂x[i, j, k, l] * xμ * idenom²

    if γ !== nothing
        @inbounds ∂γ[i, j, k, 1] = ∂y[i, j, k, l] * xμ * idenom
    end
end

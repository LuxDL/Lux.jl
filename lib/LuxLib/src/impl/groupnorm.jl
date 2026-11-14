groupnorm_reduce_dims(::AbstractArray{T,N}) where {T,N} = ntuple(static, N - 1)

CRC.@non_differentiable groupnorm_reduce_dims(::Any)

function groupnorm(
    x::AbstractArray{xT,N},
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    groups::Int,
    act::F,
    ϵ,
) where {F,N,xT}
    x′ = Utils.safe_reshape(
        x, size(x)[1:(N - 2)]..., size(x, N - 1) ÷ groups, groups, size(x, N)
    )
    (μ, σ²), _ = compute_batch_statistics(
        x′, nothing, nothing, groupnorm_reduce_dims(x), False(), nothing
    )
    return Utils.safe_reshape(
        groupnorm_affine_normalize(act, x′, μ, σ², γ, β, ϵ), size(x)...
    )
end

function groupnorm_affine_normalize(
    act::F,
    x::AbstractArray{xT,N},
    μ::AbstractArray{μT,N},
    σ²::AbstractArray{σ²T,N},
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
) where {F,N,xT,μT,σ²T}
    return groupnorm_affine_normalize(
        internal_operation_mode((x, μ, σ², γ, β)), act, x, μ, σ², γ, β, ϵ
    )
end

function groupnorm_affine_normalize(
    ::GenericBroadcastOp,
    act::F,
    x::AbstractArray{xT,N},
    μ::AbstractArray{μT,N},
    σ²::AbstractArray{σ²T,N},
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
) where {F,N,xT,μT,σ²T}
    return affine_normalize(
        act, x, μ, σ², reshape_norm_dims(γ, size(x)), reshape_norm_dims(β, size(x)), ϵ
    )
end

@generated function groupnorm_affine_normalize(
    opmode::AbstractInternalArrayOpMode,
    act::F,
    x::AbstractArray{xT,N},
    μ::AbstractArray{μT,N},
    σ²::AbstractArray{σ²T,N},
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
) where {F,N,xT,μT,σ²T}
    reshape_calls = if γ != Nothing
        quote
            γ′ = Utils.safe_reshape(γ, 1, size(x, N - 2), size(x, N - 1), 1)
            β′ = Utils.safe_reshape(β, 1, size(x, N - 2), size(x, N - 1), 1)
        end
    else
        quote
            γ′ = nothing
            β′ = nothing
        end
    end

    return quote
        x′ = Utils.safe_reshape(x, :, size(x, N - 2), size(x, N - 1), size(x, N))
        μ′ = Utils.safe_reshape(μ, 1, 1, size(x, N - 1), size(x, N))
        σ²′ = Utils.safe_reshape(σ², 1, 1, size(x, N - 1), size(x, N))
        $(reshape_calls)
        return Utils.safe_reshape(
            groupnorm_affine_normalize_internal(opmode, act, x′, μ′, σ²′, γ′, β′, ϵ),
            size(x)...,
        )
    end
end

@stable default_mode = "disable" function groupnorm_affine_normalize_internal(
    opmode::AbstractInternalArrayOpMode,
    act::F,
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {F,xT,μT,σ²T}
    y = similar(
        x,
        promote_type(
            safe_eltype(x), safe_eltype(μ), safe_eltype(σ²), safe_eltype(γ), safe_eltype(β)
        ),
    )
    groupnorm_affine_normalize_internal!(y, opmode, act, x, μ, σ², γ, β, ϵ)
    return y
end

function groupnorm_affine_normalize_internal!(
    y::AbstractArray{yT,4},
    opmode::LoopedArrayOp,
    act::F,
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {F,xT,yT,μT,σ²T}
    if unsafe_known(fuse_cpu_activation(act))
        groupnorm_affine_normalize_act_cpu!(y, x, μ, σ², γ, β, ϵ, act)
    else
        groupnorm_affine_normalize_cpu!(y, x, μ, σ², γ, β, ϵ)
        activation!(y, opmode, act, y)
    end
    return nothing
end

function groupnorm_affine_normalize_act_cpu!(
    y::AbstractArray{yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
    act::F,
) where {F,xT,yT,μT,σ²T}
    return if size(y, 1) == 1
        groupnorm_affine_normalize_act_3d_serial_cpu!(y, x, μ, σ², γ, β, ϵ, act)
    else
        groupnorm_affine_normalize_act_4d_serial_cpu!(y, x, μ, σ², γ, β, ϵ, act)
    end
end

function groupnorm_affine_normalize_act_3d_serial_cpu!(
    y::AbstractArray{yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
    σ::F,
) where {F,xT,yT,μT,σ²T}
    return if γ === nothing && β === nothing
        @fastmath @inbounds for L in axes(y, 4), K in axes(y, 3)
            γ′ = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            β′ = -μ[1, 1, K, L] * γ′
            @simd ivdep for J in axes(y, 2)
                y[1, J, K, L] = σ(x[1, J, K, L] * γ′ + β′)
            end
        end
    else
        @fastmath @inbounds for L in axes(y, 4), K in axes(y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            @simd for J in axes(y, 2)
                γ′ = γ[1, J, K, 1] * idenom
                β′ = β[1, J, K, 1] - μ[1, 1, K, L] * γ′
                y[1, J, K, L] = σ(x[1, J, K, L] * γ′ + β′)
            end
        end
    end
end

function groupnorm_affine_normalize_act_4d_serial_cpu!(
    y::AbstractArray{yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
    σ::F,
) where {F,xT,yT,μT,σ²T}
    return if γ === nothing && β === nothing
        @fastmath @inbounds for L in axes(y, 4), K in axes(y, 3)
            γ′ = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            β′ = -μ[1, 1, K, L] * γ′
            for J in axes(y, 2)
                @simd ivdep for I in axes(y, 1)
                    y[I, J, K, L] = σ(x[I, J, K, L] * γ′ + β′)
                end
            end
        end
    else
        @fastmath @inbounds for L in axes(y, 4), K in axes(y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            for J in axes(y, 2)
                γ′ = γ[1, J, K, 1] * idenom
                β′ = β[1, J, K, 1] - μ[1, 1, K, L] * γ′
                @simd ivdep for I in axes(y, 1)
                    y[I, J, K, L] = σ(x[I, J, K, L] * γ′ + β′)
                end
            end
        end
    end
end

function groupnorm_affine_normalize_cpu!(
    y::AbstractArray{yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {xT,yT,μT,σ²T}
    return if size(y, 1) == 1
        groupnorm_affine_normalize_3d_serial_cpu!(y, x, μ, σ², γ, β, ϵ)
    else
        groupnorm_affine_normalize_4d_serial_cpu!(y, x, μ, σ², γ, β, ϵ)
    end
end

@inline function groupnorm_affine_normalize_3d_serial_cpu!(
    y::AbstractArray{yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {xT,yT,μT,σ²T}
    return if γ === nothing && β === nothing
        @fastmath @inbounds for L in axes(y, 4), K in axes(y, 3)
            γ′ = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            β′ = -μ[1, 1, K, L] * γ′
            @simd ivdep for J in axes(y, 2)
                y[1, J, K, L] = x[1, J, K, L] * γ′ + β′
            end
        end
    else
        @fastmath @inbounds for L in axes(y, 4), K in axes(y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            @simd for J in axes(y, 2)
                γ′ = γ[1, J, K, 1] * idenom
                β′ = β[1, J, K, 1] - μ[1, 1, K, L] * γ′
                y[1, J, K, L] = x[1, J, K, L] * γ′ + β′
            end
        end
    end
end

@inline function groupnorm_affine_normalize_4d_serial_cpu!(
    y::AbstractArray{yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {xT,yT,μT,σ²T}
    return if γ === nothing && β === nothing
        @fastmath @inbounds for L in axes(y, 4), K in axes(y, 3)
            γ′ = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            β′ = -μ[1, 1, K, L] * γ′
            for J in axes(y, 2)
                @simd ivdep for I in axes(y, 1)
                    y[I, J, K, L] = x[I, J, K, L] * γ′ + β′
                end
            end
        end
    else
        @fastmath @inbounds for L in axes(y, 4), K in axes(y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            for J in axes(y, 2)
                γ′ = γ[1, J, K, 1] * idenom
                β′ = β[1, J, K, 1] - μ[1, 1, K, L] * γ′
                @simd ivdep for I in axes(y, 1)
                    y[I, J, K, L] = x[I, J, K, L] * γ′ + β′
                end
            end
        end
    end
end

function groupnorm_affine_normalize_internal!(
    y::AbstractArray{yT,4},
    ::GPUBroadcastOp,
    act::F,
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {F,xT,yT,μT,σ²T}
    backend = KA.get_backend(y)
    run_ka_kernel(
        groupnorm_affine_normalize_kernel!,
        backend,
        nothing,
        size(y),
        y,
        act,
        x,
        μ,
        σ²,
        γ,
        β,
        ϵ,
    )
    return nothing
end

@kernel cpu = false inbounds = true function groupnorm_affine_normalize_kernel!(
    y::AbstractArray{<:Number,4},
    @Const(f),
    @Const(x),
    @Const(μ),
    @Const(σ²),
    @Const(γ::Nothing),
    @Const(β::Nothing),
    @Const(ϵ)
)
    i, j, k, l = @index(Global, NTuple)
    γ′ = inv(sqrt(σ²[1, 1, k, l] + ϵ))
    β′ = -μ[1, 1, k, l] * γ′
    y[i, j, k, l] = f(muladd(x[i, j, k, l], γ′, β′))
    return nothing
end

@kernel cpu = false inbounds = true function groupnorm_affine_normalize_kernel!(
    y::AbstractArray{<:Number,4},
    @Const(f),
    @Const(x),
    @Const(μ),
    @Const(σ²),
    @Const(γ),
    @Const(β),
    @Const(ϵ)
)
    i, j, k, l = @index(Global, NTuple)
    γ′ = γ[1, j, k, 1] / sqrt(σ²[1, 1, k, l] + ϵ)
    β′ = muladd(-μ[1, 1, k, l], γ′, β[1, j, k, 1])
    y[i, j, k, l] = f(muladd(x[i, j, k, l], γ′, β′))
    return nothing
end

function CRC.rrule(
    cfg::RuleConfig{>:HasReverseMode},
    ::typeof(groupnorm_affine_normalize_internal),
    opmode::AbstractInternalArrayOpMode,
    f::F,
    x::AbstractArray{T,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {F,T,μT,σ²T}
    y = similar(
        x,
        promote_type(
            safe_eltype(x), safe_eltype(μ), safe_eltype(σ²), safe_eltype(γ), safe_eltype(β)
        ),
    )
    groupnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ)
    z, ∇activation = CRC.rrule_via_ad(cfg, activation!!, opmode, is_mutable_array(y), f, y)

    𝒫x, 𝒫μ, 𝒫σ² = CRC.ProjectTo(x), CRC.ProjectTo(μ), CRC.ProjectTo(σ²)
    𝒫γ = γ === nothing ? identity : CRC.ProjectTo(γ)
    𝒫β = β === nothing ? identity : CRC.ProjectTo(β)

    ∇groupnorm_affine_normalize_internal = @closure Δ -> begin
        ∂y = recursive_unthunk(last(∇activation(Δ)))
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇groupnorm_affine_normalize(opmode, ∂y, x, μ, σ², γ, β, ϵ)
        return ∂∅, ∂∅, ∂∅, 𝒫x(∂x), 𝒫μ(∂μ), 𝒫σ²(∂σ²), 𝒫γ(∂γ), 𝒫β(∂β), ∂∅
    end

    return z, ∇groupnorm_affine_normalize_internal
end

function ∇groupnorm_affine_normalize(
    opmode::AbstractInternalArrayOpMode,
    ∂y::AbstractArray{∂yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {∂yT,xT,μT,σ²T}
    ∂x, ∂σ² = similar(x), similar(σ², size(x))
    ∂γ = γ === nothing ? nothing : similar(γ, size(x))

    ∇groupnorm_affine_normalize!(∂x, ∂σ², ∂γ, opmode, ∂y, x, μ, σ², γ, ϵ)

    ∂μ = sum(-, ∂x; dims=(1, 2))
    ∂σ² = sum(∂σ²; dims=(1, 2))
    ∂γ = γ === nothing ? ∂∅ : sum(∂γ; dims=(1, 4))
    ∂β = β === nothing ? ∂∅ : sum(∂y; dims=(1, 4))

    return ∂x, ∂μ, ∂σ², ∂γ, ∂β
end

function ∇groupnorm_affine_normalize(
    ::LoopedArrayOp,
    ∂y::AbstractArray{∂yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    β::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {∂yT,xT,μT,σ²T}
    ∂x, ∂μ, ∂σ² = similar(x), similar(μ), similar(σ²)
    ∂γ = γ === nothing ? nothing : similar(γ)
    ∂β = β === nothing ? nothing : similar(β)

    ∇groupnorm_affine_normalize_cpu!(∂x, ∂μ, ∂σ², ∂γ, ∂β, ∂y, x, μ, σ², γ, ϵ)

    ∂γ = γ === nothing ? ∂∅ : ∂γ
    ∂β = β === nothing ? ∂∅ : ∂β

    return ∂x, ∂μ, ∂σ², ∂γ, ∂β
end

function ∇groupnorm_affine_normalize_cpu!(
    ∂x::AbstractArray{∂xT,4},
    ∂μ::AbstractArray{∂μT,4},
    ∂σ²::AbstractArray{∂σ²T,4},
    ::Nothing,
    ::Nothing,
    ∂y::AbstractArray{∂yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    ::Nothing,
    ϵ,
) where {∂xT,∂μT,∂σ²T,∂yT,xT,μT,σ²T}
    half = eltype(∂σ²)(0.5)

    fill!(∂μ, 0)
    fill!(∂σ², 0)

    return if size(∂y, 1) == 1
        @fastmath @inbounds for L in axes(∂y, 4), K in axes(∂y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            idenom² = idenom^2

            @simd for J in axes(∂y, 2)
                xμ = x[1, J, K, L] - μ[1, 1, K, L]

                ∂x[1, J, K, L] = ∂y[1, J, K, L] * idenom
                ∂μ[1, 1, K, L] -= ∂x[1, J, K, L]
                ∂σ²[1, 1, K, L] -= ∂x[1, J, K, L] * xμ * half * idenom²
            end
        end
    else
        @fastmath @inbounds for L in axes(∂y, 4), K in axes(∂y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            idenom² = idenom^2

            for J in axes(∂y, 2)
                @simd for I in axes(∂y, 1)
                    xμ = x[I, J, K, L] - μ[1, 1, K, L]

                    ∂x[I, J, K, L] = ∂y[I, J, K, L] * idenom
                    ∂μ[1, 1, K, L] -= ∂x[I, J, K, L]
                    ∂σ²[1, 1, K, L] -= ∂x[I, J, K, L] * xμ * half * idenom²
                end
            end
        end
    end
end

function ∇groupnorm_affine_normalize_cpu!(
    ∂x::AbstractArray{∂xT,4},
    ∂μ::AbstractArray{∂μT,4},
    ∂σ²::AbstractArray{∂σ²T,4},
    ∂γ::AbstractArray{∂γT,4},
    ∂β::AbstractArray{∂βT,4},
    ∂y::AbstractArray{∂yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::AbstractArray{γT,4},
    ϵ,
) where {∂xT,∂μT,∂σ²T,∂γT,∂βT,∂yT,xT,μT,σ²T,γT}
    half = eltype(∂σ²)(0.5)

    fill!(∂μ, 0)
    fill!(∂σ², 0)
    fill!(∂γ, 0)
    fill!(∂β, 0)

    return if size(∂y, 1) == 1
        @fastmath @inbounds for L in axes(∂y, 4), K in axes(∂y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            idenom² = idenom^2

            @simd for J in axes(∂y, 2)
                γ′ = γ[1, J, K, 1] * idenom

                xμ = x[1, J, K, L] - μ[1, 1, K, L]

                ∂x[1, J, K, L] = ∂y[1, J, K, L] * γ′
                ∂μ[1, 1, K, L] -= ∂x[1, J, K, L]
                ∂σ²[1, 1, K, L] -= ∂x[1, J, K, L] * xμ * half * idenom²
                ∂γ[1, J, K, 1] += ∂y[1, J, K, L] * xμ * idenom
                ∂β[1, J, K, 1] += ∂y[1, J, K, L]
            end
        end
    else
        @fastmath @inbounds for L in axes(∂y, 4), K in axes(∂y, 3)
            idenom = inv(sqrt(σ²[1, 1, K, L] + ϵ))
            idenom² = idenom^2

            for J in axes(∂y, 2)
                γ′ = γ[1, J, K, 1] * idenom
                @simd for I in axes(∂y, 1)
                    xμ = x[I, J, K, L] - μ[1, 1, K, L]

                    ∂x[I, J, K, L] = ∂y[I, J, K, L] * γ′
                    ∂μ[1, 1, K, L] -= ∂x[I, J, K, L]
                    ∂σ²[1, 1, K, L] -= ∂x[I, J, K, L] * xμ * half * idenom²
                    ∂γ[1, J, K, 1] += ∂y[I, J, K, L] * xμ * idenom
                    ∂β[1, J, K, 1] += ∂y[I, J, K, L]
                end
            end
        end
    end
end

function ∇groupnorm_affine_normalize!(
    ∂x::AbstractArray{∂xT,4},
    ∂σ²::AbstractArray{∂σ²T,4},
    ∂γ::Optional{<:AbstractArray{<:Any,4}},
    ::GPUBroadcastOp,
    ∂y::AbstractArray{∂yT,4},
    x::AbstractArray{xT,4},
    μ::AbstractArray{μT,4},
    σ²::AbstractArray{σ²T,4},
    γ::Optional{<:AbstractArray{<:Any,4}},
    ϵ,
) where {∂xT,∂σ²T,∂yT,xT,μT,σ²T}
    backend = KA.get_backend(∂x)
    run_ka_kernel(
        ∇groupnorm_affine_normalize_kernel!,
        backend,
        nothing,
        size(∂x),
        ∂x,
        ∂σ²,
        ∂γ,
        ∂y,
        x,
        μ,
        σ²,
        ϵ,
        γ,
    )
    return nothing
end

@kernel cpu = false inbounds = true function ∇groupnorm_affine_normalize_kernel!(
    ∂x,
    ∂σ²,
    @Const(∂γ::Nothing),
    @Const(∂y),
    @Const(x),
    @Const(μ),
    @Const(σ²),
    @Const(ϵ),
    @Const(γ::Nothing)
)
    i, j, k, l = @index(Global, NTuple)
    idenom = inv(sqrt(σ²[1, 1, k, l] + ϵ))

    ∂x[i, j, k, l] = ∂y[i, j, k, l] * idenom
    ∂σ²[i, j, k, l] = ∂x[i, j, k, l] * (μ[1, 1, k, l] - x[i, j, k, l]) * idenom * idenom / 2
    return nothing
end

@kernel cpu = false inbounds = true function ∇groupnorm_affine_normalize_kernel!(
    ∂x, ∂σ², ∂γ, @Const(∂y), @Const(x), @Const(μ), @Const(σ²), @Const(ϵ), @Const(γ)
)
    i, j, k, l = @index(Global, NTuple)
    idenom = inv(sqrt(σ²[1, 1, k, l] + ϵ))
    γ′ = γ[1, j, k, 1] * idenom

    xμ_d = (x[i, j, k, l] - μ[1, 1, k, l]) * idenom

    ∂x[i, j, k, l] = ∂y[i, j, k, l] * γ′
    ∂σ²[i, j, k, l] = -∂x[i, j, k, l] * xμ_d * idenom / 2
    ∂γ[i, j, k, l] = ∂y[i, j, k, l] * xμ_d
    return nothing
end

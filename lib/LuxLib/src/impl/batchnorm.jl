function batchnorm_cudnn end   # Defined in LuxLibcuDNNExt
function ∇batchnorm_cudnn end  # Defined in LuxLibcuDNNExt

function batchnorm_reduce_dims(::AbstractArray{T,N}) where {T,N}
    return (ntuple(static, N - 2)..., static(N))
end

CRC.@non_differentiable batchnorm_reduce_dims(::Any...)

function get_batchnorm_statistics(
    ::AbstractArray, rμ::Optional{<:AbstractVector}, rσ²::Optional{<:AbstractVector}, ::True
)
    return copy_drop_gradients(rμ), copy_drop_gradients(rσ²)
end

function get_batchnorm_statistics(x::AbstractArray, ::Nothing, ::Nothing, ::False)
    μ, σ² = mean_var(x; dims=unsafe_known(batchnorm_reduce_dims(x)), corrected=false)
    return safe_vec(μ), safe_vec(σ²)
end

function get_batchnorm_statistics(
    ::AbstractArray, rμ::AbstractVector, rσ²::AbstractVector, ::False
)
    return rμ, rσ²
end

CRC.@non_differentiable get_batchnorm_statistics(::Any...)

function batchnorm(
    x::AbstractArray{xT,N},
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    rμ::Optional{<:AbstractVector},
    rσ²::Optional{<:AbstractVector},
    training::StaticBool,
    act::F,
    momentum,
    ϵ,
) where {F,xT,N}
    (μ, σ²), (rμ, rσ²) = compute_batch_statistics(
        x,
        reshape_norm_dims(rμ, size(x)),
        reshape_norm_dims(rσ², size(x)),
        batchnorm_reduce_dims(x),
        training,
        momentum,
    )
    return batchnorm_affine_normalize(act, x, μ, σ², γ, β, ϵ), safe_vec(rμ), safe_vec(rσ²)
end

function batchnorm_affine_normalize(
    act::F,
    x::AbstractArray{xT,N},
    μ::AbstractArray{μT,N},
    σ²::AbstractArray{σ²T,N},
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
) where {F,xT,μT,σ²T,N}
    return batchnorm_affine_normalize(
        internal_operation_mode((x, μ, σ², γ, β)), act, x, μ, σ², γ, β, ϵ
    )
end

function batchnorm_affine_normalize(
    ::GenericBroadcastOp,
    act::F,
    x::AbstractArray{xT,N},
    μ::AbstractArray{μT,N},
    σ²::AbstractArray{σ²T,N},
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
) where {F,xT,μT,σ²T,N}
    return affine_normalize(
        act, x, μ, σ², reshape_norm_dims(γ, size(x)), reshape_norm_dims(β, size(x)), ϵ
    )
end

function batchnorm_affine_normalize(
    opmode::AbstractInternalArrayOpMode,
    act::F,
    x::AbstractArray{xT,N},
    μ::AbstractArray{μT,N},
    σ²::AbstractArray{σ²T,N},
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
) where {F,xT,μT,σ²T,N}
    x′ = Utils.safe_reshape(x, :, size(x, N - 1), size(x, N))
    return Utils.safe_reshape(
        batchnorm_affine_normalize_internal(opmode, act, x′, vec(μ), vec(σ²), γ, β, ϵ),
        size(x)...,
    )
end

@stable default_mode = "disable" function batchnorm_affine_normalize_internal(
    opmode::AbstractInternalArrayOpMode,
    act::F,
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
) where {F,xT}
    y = similar(
        x,
        promote_type(
            safe_eltype(x), safe_eltype(μ), safe_eltype(σ²), safe_eltype(γ), safe_eltype(β)
        ),
    )
    batchnorm_affine_normalize_internal!(y, opmode, act, x, μ, σ², γ, β, ϵ)
    return y
end

function batchnorm_affine_normalize_internal!(
    y::AbstractArray{yT,3},
    opmode::LoopedArrayOp,
    act::F,
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
    γ′::Optional{<:AbstractVector}=nothing,
) where {F,xT,yT}
    N = size(y, 2)
    γ′ = if γ′ === nothing
        similar(x, promote_type(safe_eltype(γ), safe_eltype(σ²), safe_eltype(ϵ)), N)
    else
        γ′
    end
    β′ = similar(x, promote_type(safe_eltype(β), safe_eltype(σ²), safe_eltype(ϵ)), N)

    compute_batchnorm_scale_bias!(γ′, β′, γ, β, μ, σ², ϵ)

    if unsafe_known(fuse_cpu_activation(act))
        apply_batchnorm_scale_bias_act_cpu!(y, γ′, β′, x, act)
    else
        apply_batchnorm_scale_bias_cpu!(y, γ′, β′, x)
        activation!(y, opmode, act, y)
    end

    return nothing
end

function compute_batchnorm_scale_bias!(γ′, β′, γ, β, μ, σ², ϵ)
    if Utils.within_enzyme_autodiff()
        if γ === nothing && β === nothing
            for J in eachindex(γ′, β′, μ, σ²)
                @inbounds γ′[J] = inv(sqrt(σ²[J] + ϵ))
                @inbounds β′[J] = -μ[J] * γ′[J]
            end
        else
            for J in eachindex(γ′, β′, γ, β, μ, σ²)
                @inbounds γ′[J] = γ[J] / sqrt(σ²[J] + ϵ)
                @inbounds β′[J] = β[J] - μ[J] * γ′[J]
            end
        end
    else
        if γ === nothing && β === nothing
            @simd ivdep for J in eachindex(γ′, β′, μ, σ²)
                @fastmath @inbounds γ′[J] = inv(sqrt(σ²[J] + ϵ))
                @fastmath @inbounds β′[J] = -μ[J] * γ′[J]
            end
        else
            @simd ivdep for J in eachindex(γ′, β′, γ, β, μ, σ²)
                @fastmath @inbounds γ′[J] = γ[J] / sqrt(σ²[J] + ϵ)
                @fastmath @inbounds β′[J] = β[J] - μ[J] * γ′[J]
            end
        end
    end
    return nothing
end

function apply_batchnorm_scale_bias_act_cpu!(
    y::AbstractArray{yT,3},
    γ′::AbstractVector,
    β′::AbstractVector,
    x::AbstractArray{xT,3},
    σ::F,
) where {F,xT,yT}
    if size(y, 1) == 1
        apply_batchnorm_scale_bias_act_2d_serial_cpu!(y, γ′, β′, x, σ)
    else
        if Utils.within_enzyme_autodiff()
            apply_batchnorm_scale_bias_act_3d_serial_cpu!(y, γ′, β′, x, σ)
        else
            apply_batchnorm_scale_bias_act_3d_threaded_cpu!(y, γ′, β′, x, σ)
        end
    end
    return nothing
end

@inline function apply_batchnorm_scale_bias_act_2d_serial_cpu!(
    y::AbstractArray{yT,3},
    γ′::AbstractVector,
    β′::AbstractVector,
    x::AbstractArray{xT,3},
    σ::F,
) where {F,xT,yT}
    for K in axes(x, 3)
        @simd ivdep for J in axes(x, 2)
            @fastmath @inbounds y[1, J, K] = σ(x[1, J, K] * γ′[J] + β′[J])
        end
    end
    return nothing
end

@inline function apply_batchnorm_scale_bias_act_3d_threaded_cpu!(
    y::AbstractArray{yT,3},
    γ′::AbstractVector,
    β′::AbstractVector,
    x::AbstractArray{xT,3},
    σ::F,
) where {F,xT,yT}
    Threads.@threads for K in axes(x, 3)
        for J in axes(x, 2)
            @simd ivdep for I in axes(x, 1)
                @fastmath @inbounds y[I, J, K] = σ(x[I, J, K] * γ′[J] + β′[J])
            end
        end
    end
    return nothing
end

@inline function apply_batchnorm_scale_bias_act_3d_serial_cpu!(
    y::AbstractArray{yT,3},
    γ′::AbstractVector,
    β′::AbstractVector,
    x::AbstractArray{xT,3},
    σ::F,
) where {F,xT,yT}
    for K in axes(x, 3)
        for J in axes(x, 2)
            @simd ivdep for I in axes(x, 1)
                @fastmath @inbounds y[I, J, K] = σ(x[I, J, K] * γ′[J] + β′[J])
            end
        end
    end
    return nothing
end

@enzyme_alternative apply_batchnorm_scale_bias_act_3d_threaded_cpu! apply_batchnorm_scale_bias_act_3d_serial_cpu!

function apply_batchnorm_scale_bias_cpu!(
    y::AbstractArray{yT,3}, γ′::AbstractVector, β′::AbstractVector, x::AbstractArray{xT,3}
) where {xT,yT}
    if size(y, 1) == 1
        apply_batchnorm_scale_bias_2d_serial_cpu!(y, γ′, β′, x)
    else
        if Utils.within_enzyme_autodiff()
            apply_batchnorm_scale_bias_3d_serial_cpu!(y, γ′, β′, x)
        else
            apply_batchnorm_scale_bias_3d_threaded_cpu!(y, γ′, β′, x)
        end
    end
    return nothing
end

@inline function apply_batchnorm_scale_bias_2d_serial_cpu!(
    y::AbstractArray{yT,3}, γ′::AbstractVector, β′::AbstractVector, x::AbstractArray{xT,3}
) where {xT,yT}
    for K in axes(x, 3)
        @simd ivdep for J in axes(x, 2)
            @fastmath @inbounds y[1, J, K] = x[1, J, K] * γ′[J] + β′[J]
        end
    end
    return nothing
end

@inline function apply_batchnorm_scale_bias_3d_threaded_cpu!(
    y::AbstractArray{yT,3}, γ′::AbstractVector, β′::AbstractVector, x::AbstractArray{xT,3}
) where {xT,yT}
    Threads.@threads for K in axes(x, 3)
        for J in axes(x, 2)
            @simd ivdep for I in axes(x, 1)
                @fastmath @inbounds y[I, J, K] = x[I, J, K] * γ′[J] + β′[J]
            end
        end
    end
    return nothing
end

@inline function apply_batchnorm_scale_bias_3d_serial_cpu!(
    y::AbstractArray{yT,3}, γ′::AbstractVector, β′::AbstractVector, x::AbstractArray{xT,3}
) where {xT,yT}
    for K in axes(x, 3)
        for J in axes(x, 2)
            @simd ivdep for I in axes(x, 1)
                @fastmath @inbounds y[I, J, K] = x[I, J, K] * γ′[J] + β′[J]
            end
        end
    end
    return nothing
end

@enzyme_alternative apply_batchnorm_scale_bias_3d_threaded_cpu! apply_batchnorm_scale_bias_3d_serial_cpu!

function batchnorm_affine_normalize_internal!(
    y::AbstractArray{yT,3},
    ::GPUBroadcastOp,
    act::F,
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
    γ′::Optional{<:AbstractVector}=nothing,
) where {F,xT,yT}
    backend = KA.get_backend(y)
    run_ka_kernel(
        batchnorm_affine_normalize_internal_kernel!,
        backend,
        nothing,
        size(y),
        y,
        γ′,
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

@kernel cpu = false inbounds = true function batchnorm_affine_normalize_internal_kernel!(
    y::AbstractArray{<:Number,3},
    @Const(γ′::Nothing),
    @Const(f),
    @Const(x),
    @Const(μ),
    @Const(σ²),
    @Const(γ::Nothing),
    @Const(β::Nothing),
    @Const(ϵ)
)
    i, j, k = @index(Global, NTuple)
    γ′′ = inv(sqrt(σ²[j] + ϵ))
    β′ = -μ[j] * γ′′
    y[i, j, k] = f(muladd(x[i, j, k], γ′′, β′))
    return nothing
end

@kernel cpu = false inbounds = true function batchnorm_affine_normalize_internal_kernel!(
    y::AbstractArray{<:Number,3},
    γ′::AbstractVector{<:Number},
    @Const(f),
    @Const(x),
    @Const(μ),
    @Const(σ²),
    @Const(γ::Nothing),
    @Const(β::Nothing),
    @Const(ϵ)
)
    i, j, k = @index(Global, NTuple)
    γ′′ = inv(sqrt(σ²[j] + ϵ))
    β′ = -μ[j] * γ′′
    y[i, j, k] = f(muladd(x[i, j, k], γ′′, β′))
    if i == 1 && k == 1
        γ′[j] = γ′′
    end
    return nothing
end

@kernel cpu = false inbounds = true function batchnorm_affine_normalize_internal_kernel!(
    y::AbstractArray{<:Number,3},
    @Const(γ′::Nothing),
    @Const(f),
    @Const(x),
    @Const(μ),
    @Const(σ²),
    @Const(γ),
    @Const(β),
    @Const(ϵ)
)
    i, j, k = @index(Global, NTuple)
    γ′′ = γ[j] / sqrt(σ²[j] + ϵ)
    β′ = muladd(-μ[j], γ′′, β[j])
    y[i, j, k] = f(muladd(x[i, j, k], γ′′, β′))
    return nothing
end

@kernel cpu = false inbounds = true function batchnorm_affine_normalize_internal_kernel!(
    y::AbstractArray{<:Number,3},
    γ′::AbstractVector{<:Number},
    @Const(f),
    @Const(x),
    @Const(μ),
    @Const(σ²),
    @Const(γ),
    @Const(β),
    @Const(ϵ)
)
    i, j, k = @index(Global, NTuple)
    γ′′ = γ[j] / sqrt(σ²[j] + ϵ)
    β′ = muladd(-μ[j], γ′′, β[j])
    y[i, j, k] = f(muladd(x[i, j, k], γ′′, β′))
    if i == 1 && k == 1
        γ′[j] = γ′′
    end
    return nothing
end

function CRC.rrule(
    cfg::RuleConfig{>:HasReverseMode},
    ::typeof(batchnorm_affine_normalize_internal),
    opmode::AbstractInternalArrayOpMode,
    act::F,
    x::AbstractArray{T,N},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
) where {F,T,N}
    y = similar(
        x,
        promote_type(
            safe_eltype(x), safe_eltype(μ), safe_eltype(σ²), safe_eltype(γ), safe_eltype(β)
        ),
    )
    γ′ = similar(
        x, promote_type(safe_eltype(γ), safe_eltype(σ²), safe_eltype(ϵ)), size(x, N - 1)
    )

    batchnorm_affine_normalize_internal!(y, opmode, identity, x, μ, σ², γ, β, ϵ, γ′)
    z, ∇activation = CRC.rrule_via_ad(
        cfg, activation!!, opmode, is_mutable_array(y), act, y
    )

    𝒫x, 𝒫μ, 𝒫σ² = CRC.ProjectTo(x), CRC.ProjectTo(μ), CRC.ProjectTo(σ²)
    𝒫γ = γ === nothing ? identity : CRC.ProjectTo(γ)
    𝒫β = β === nothing ? identity : CRC.ProjectTo(β)

    ∇batchnorm_affine_normalize_internal = @closure Δ -> begin
        ∂y = recursive_unthunk(last(∇activation(Δ)))
        ∂x, ∂μ, ∂σ², ∂γ, ∂β = ∇batchnorm_affine_normalize(opmode, ∂y, x, μ, σ², γ, β, ϵ, γ′)
        return ∂∅, ∂∅, ∂∅, 𝒫x(∂x), 𝒫μ(∂μ), 𝒫σ²(∂σ²), 𝒫γ(∂γ), 𝒫β(∂β), ∂∅
    end

    return z, ∇batchnorm_affine_normalize_internal
end

function ∇batchnorm_affine_normalize(
    opmode::LoopedArrayOp,
    ∂y::AbstractArray{∂yT,3},
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
    γ′::AbstractVector,
) where {∂yT,xT}
    ∂x, ∂μ, ∂σ² = similar(x), similar(μ), similar(σ²)
    ∂γ = γ === nothing ? nothing : similar(γ)
    ∂β = β === nothing ? nothing : similar(β)

    ∇batchnorm_affine_normalize_cpu!(∂x, ∂μ, ∂σ², ∂γ, ∂β, ∂y, x, μ, σ², γ, ϵ, γ′)

    ∂γ = γ === nothing ? ∂∅ : ∂γ
    ∂β = β === nothing ? ∂∅ : ∂β

    return ∂x, ∂μ, ∂σ², ∂γ, ∂β
end

function ∇batchnorm_affine_normalize_cpu!(
    ∂x::AbstractArray{∂xT,3},
    ∂μ::AbstractVector{∂μT},
    ∂σ²::AbstractVector{∂σ²T},
    ::Nothing,
    ::Nothing,
    ∂y::AbstractArray{∂yT,3},
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    ::Nothing,
    ϵ,
    γ′::AbstractVector,
) where {∂xT,∂μT,∂σ²T,∂yT,xT}
    half = eltype(∂σ²)(0.5)

    fill!(∂μ, 0)
    fill!(∂σ², 0)

    if size(∂y, 1) == 1
        @fastmath @inbounds for K in axes(∂y, 3)
            @simd for J in axes(∂y, 2)
                idenom = γ′[J]
                idenom² = idenom^2

                xμ = x[1, J, K] - μ[J]

                ∂x[1, J, K] = ∂y[1, J, K] * idenom
                ∂μ[J] -= ∂x[1, J, K]
                ∂σ²[J] -= ∂x[1, J, K] * xμ * half * idenom²
            end
        end
    else
        @fastmath @inbounds for K in axes(∂y, 3), J in axes(∂y, 2)
            idenom = γ′[J]
            idenom² = idenom^2

            @simd for I in axes(∂y, 1)
                xμ = x[I, J, K] - μ[J]

                ∂x[I, J, K] = ∂y[I, J, K] * idenom
                ∂μ[J] -= ∂x[I, J, K]
                ∂σ²[J] -= ∂x[I, J, K] * xμ * half * idenom²
            end
        end
    end
    return nothing
end

function ∇batchnorm_affine_normalize_cpu!(
    ∂x::AbstractArray{∂xT,3},
    ∂μ::AbstractVector{∂μT},
    ∂σ²::AbstractVector{∂σ²T},
    ∂γ::AbstractVector{∂γT},
    ∂β::AbstractVector{∂βT},
    ∂y::AbstractArray{∂yT,3},
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::AbstractVector,
    ϵ,
    γ′::AbstractVector,
) where {∂xT,∂μT,∂σ²T,∂γT,∂βT,∂yT,xT}
    half = eltype(∂σ²)(0.5)

    fill!(∂μ, 0)
    fill!(∂σ², 0)
    fill!(∂γ, 0)
    fill!(∂β, 0)

    if size(∂y, 1) == 1
        @fastmath @inbounds for K in axes(∂y, 3)
            @simd for J in axes(∂y, 2)
                idenom = inv(sqrt(σ²[J] + ϵ))
                idenom² = idenom^2

                xμ = x[1, J, K] - μ[J]

                ∂x[1, J, K] = ∂y[1, J, K] * γ′[J]
                ∂μ[J] -= ∂x[1, J, K]
                ∂σ²[J] -= ∂x[1, J, K] * xμ * half * idenom²
                ∂γ[J] += ∂y[1, J, K] * xμ * idenom
                ∂β[J] += ∂y[1, J, K]
            end
        end
    else
        @fastmath @inbounds for K in axes(∂y, 3), J in axes(∂y, 2)
            idenom = inv(sqrt(σ²[J] + ϵ))
            idenom² = idenom^2

            @simd for I in axes(∂y, 1)
                xμ = x[I, J, K] - μ[J]

                ∂x[I, J, K] = ∂y[I, J, K] * γ′[J]
                ∂μ[J] -= ∂x[I, J, K]
                ∂σ²[J] -= ∂x[I, J, K] * xμ * half * idenom²
                ∂γ[J] += ∂y[I, J, K] * xμ * idenom
                ∂β[J] += ∂y[I, J, K]
            end
        end
    end
    return nothing
end

function ∇batchnorm_affine_normalize(
    opmode::AbstractInternalArrayOpMode,
    ∂y::AbstractArray{∂yT,3},
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::Optional{<:AbstractVector},
    β::Optional{<:AbstractVector},
    ϵ,
    γ′::AbstractVector,
) where {∂yT,xT}
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
    ∂x::AbstractArray{∂xT,3},
    ∂σ²::AbstractArray{∂σ²T,3},
    ∂γ::Optional{<:AbstractArray{<:Any,3}},
    ::GPUBroadcastOp,
    ∂y::AbstractArray{∂yT,3},
    x::AbstractArray{xT,3},
    μ::AbstractVector,
    σ²::AbstractVector,
    γ::Optional{<:AbstractVector},
    ϵ,
    γ′::AbstractVector,
) where {∂xT,∂σ²T,∂yT,xT}
    backend = KA.get_backend(∂x)
    run_ka_kernel(
        ∇batchnorm_affine_normalize_kernel!,
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
        γ′,
    )
    return nothing
end

@kernel cpu = false inbounds = true function ∇batchnorm_affine_normalize_kernel!(
    ∂x,
    ∂σ²,
    @Const(∂γ::Nothing),
    @Const(∂y),
    @Const(x),
    @Const(μ),
    @Const(σ²),
    @Const(ϵ),
    @Const(γ′)
)
    i, j, k = @index(Global, NTuple)
    idenom = γ′[j]
    idenom² = idenom * idenom

    xμ = x[i, j, k] - μ[j]

    ∂x[i, j, k] = ∂y[i, j, k] * γ′[j]
    ∂σ²[i, j, k] = -∂x[i, j, k] * xμ * idenom² / 2
    return nothing
end

@kernel cpu = false inbounds = true function ∇batchnorm_affine_normalize_kernel!(
    ∂x, ∂σ², ∂γ, @Const(∂y), @Const(x), @Const(μ), @Const(σ²), @Const(ϵ), @Const(γ′)
)
    i, j, k = @index(Global, NTuple)
    idenom = inv(sqrt(σ²[j] + ϵ))
    idenom² = idenom * idenom

    xμ = x[i, j, k] - μ[j]

    ∂x[i, j, k] = ∂y[i, j, k] * γ′[j]
    ∂σ²[i, j, k] = -∂x[i, j, k] * xμ * idenom² / 2
    ∂γ[i, j, k] = ∂y[i, j, k] * xμ * idenom
    return nothing
end

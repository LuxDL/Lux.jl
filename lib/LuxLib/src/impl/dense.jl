function cublasLt_fused_dense end # Defined in `LuxLibCUDAExt`
function cublasLt_fused_dense! end # Defined in `LuxLibCUDAExt`

function fused_dense(::typeof(identity), weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector})
    return matmuladd(weight, x, b)
end

function fused_dense(act::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {F}
    return fused_dense(internal_operation_mode((weight, x, b)), act, weight, x, b)
end

function fused_dense(opmode::GenericBroadcastOp, act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    return bias_activation(opmode, act, matmul(opmode, weight, x), b)
end

@stable default_mode="disable" function fused_dense(
        opmode::AbstractInternalArrayOpMode, act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    y = similar(weight, Utils.concrete_bias_act_output_eltype(act, weight, x, b),
        size(weight, 1), size(x, 2))
    fused_dense!(y, opmode, act, weight, x, b)
    return y
end

function fused_dense!(y::AbstractMatrix, opmode::AbstractInternalArrayOpMode, act::F,
        weight::AbstractMatrix, x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    matmul!(y, opmode, weight, x)
    bias_activation!(y, opmode, act, y, b)
    return nothing
end

function fused_dense!(
        y::AbstractMatrix, ::GPUBroadcastOp{CUDADevice}, act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    cublasLt_fused_dense!(y, act, weight, x, b)
    return nothing
end

function CRC.rrule(cfg::CRC.RuleConfig{>:HasReverseMode}, ::typeof(fused_dense),
        opmode::AbstractInternalArrayOpMode, act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    T = Utils.concrete_bias_act_output_eltype(act, weight, x, b)
    𝒫weight, 𝒫x, 𝒫b = CRC.ProjectTo(weight), CRC.ProjectTo(x), CRC.ProjectTo(b)

    if Utils.known(Traits.activation_intermediate_not_needed(act, T))
        y = fused_dense(opmode, act, weight, x, b)
        ∇fused_dense_no_intermediate = @closure Δ -> begin
            ∂y = ∇activation(CRC.unthunk(Δ), y, act, Utils.NotaNumber())
            ∂w, ∂x, ∂b = ∇matmul_bias(∂y, weight, x, b)
            return ∂∅, ∂∅, ∂∅, 𝒫weight(∂w), 𝒫x(∂x), 𝒫b(∂b)
        end
        return y, ∇fused_dense_no_intermediate
    end

    if Utils.known(Traits.activation_has_rrule(act, T))
        y = matmuladd(weight, x, b)
        z = activation(opmode, act, y)
        ∇fused_dense_cached = @closure Δ -> begin
            ∂y = ∇activation(CRC.unthunk(Δ), z, act, y)
            ∂w, ∂x, ∂b = ∇matmul_bias(∂y, weight, x, b)
            return ∂∅, ∂∅, ∂∅, 𝒫weight(∂w), 𝒫x(∂x), 𝒫b(∂b)
        end
        return z, ∇fused_dense_cached
    end

    y = similar(weight, T, size(weight, 1), size(x, 2))
    matmul!(y, opmode, weight, x)
    z, ∇bias_activation = CRC.rrule_via_ad(cfg, bias_activation, opmode, act, y, b)
    ∇fused_dense_fallback = @closure Δ -> begin
        _, _, _, ∂y, ∂b = ∇bias_activation(Δ)
        ∂w, ∂x, _ = ∇matmul_bias(∂y, ∂b, weight, x, b)
        return ∂∅, ∂∅, ∂∅, 𝒫weight(∂w), 𝒫x(∂x), 𝒫b(∂b)
    end
    return z, ∇fused_dense_fallback
end

## Special Reverse Pass for gelu activation. All other cases, we don't need special handling
function CRC.rrule(
        ::typeof(fused_dense), ::GPUBroadcastOp{CUDADevice}, ::typeof(NNlib.gelu),
        weight::AbstractMatrix, x::AbstractMatrix, b::Optional{<:AbstractVector})
    z, y = cublasLt_fused_dense(NNlib.gelu, weight, x, b, True())
    𝒫weight, 𝒫x, 𝒫b = CRC.ProjectTo(weight), CRC.ProjectTo(x), CRC.ProjectTo(b)

    ∇fused_dense = @closure Δ -> begin
        ∂y = ∇activation(CRC.unthunk(Δ), z, gelu, y)
        ∂w, ∂x, ∂b = ∇matmul_bias(∂y, weight, x, b)
        return ∂∅, ∂∅, ∂∅, 𝒫weight(∂w), 𝒫x(∂x), 𝒫b(∂b)
    end

    return z, ∇fused_dense
end

∇matmul_bias(∂y, weight, x, bias) = ∇matmul_bias(∂y, ∇bias_add(bias, ∂y), weight, x, bias)
∇matmul_bias(∂y, ∂b, weight, x, _) = matmul(∂y, x'), matmul(weight', ∂y), ∂b

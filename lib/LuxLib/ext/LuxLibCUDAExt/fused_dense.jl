@inline __length(x) = length(x)
@inline __length(::Nothing) = nothing

function LuxLib.__fused_dense_bias_activation_impl(
        act::F, weight::CUDA.AnyCuMatrix, x::CUDA.AnyCuMatrix,
        b::Union{Nothing, CUDA.AnyCuVector}) where {F}
    y = similar(x, LuxLib.__get_concrete_fba_output_eltype(act, weight, x, b),
        size(weight, 1), size(x, 2))
    if hasmethod(LuxLib._cublaslt_matmul_fused!,
        (typeof(y), F, typeof(weight), typeof(x), typeof(b)))
        retcode = LuxLib._cublaslt_matmul_fused!(y, act, weight, x, b)
        retcode == 0 && return y
        # cuBLASLt failed for the given inputs use the generic fallback
        @warn "cuBLASLt failed for the given inputs $(act), $(typeof(weight)) \
               [$(size(weight))], $(typeof(x)) [$(size(x))], $(typeof(b)) \
               [$(__length(b))]. Falling back to generic implementation." maxlog=1
    else
        @warn "cuBLASLt not available. Falling back to generic implementation." maxlog=1
    end
    mul!(y, weight, x)
    return LuxLib.__apply_bias_activation!!(act, y, b, Val(false))
end

## Hijack mixed precision on CUDA to use cuBLASLt if possible
@inline function LuxLib.fused_dense_bias_activation(
        σ::F, weight::CUDA.AnyCuMatrix{wT}, x::CUDA.AnyCuMatrix{xT},
        b::CUDA.AnyCuVector{bT}) where {F, wT, xT, bT}
    return LuxLib.__fused_dense_bias_activation_impl(σ, weight, x, b)
end

@inline function LuxLib.fused_dense_bias_activation(σ::F, weight::CUDA.AnyCuMatrix{wT},
        x::CUDA.AnyCuMatrix{xT}, b::Nothing) where {F, wT, xT}
    return LuxLib.__fused_dense_bias_activation_impl(σ, weight, x, b)
end

## Special Reverse Pass for gelu activation. All other cases, we don't need special handling
function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(LuxLib.__fused_dense_bias_activation_impl), ::typeof(NNlib.gelu),
        weight::CUDA.AnyCuMatrix, x::CUDA.AnyCuMatrix, b::Union{CUDA.AnyCuVector, Nothing})
    z = similar(x, LuxLib.__get_concrete_fba_output_eltype(NNlib.gelu, weight, x, b),
        size(weight, 1), size(x, 2))
    y = z # aliased for now for type stability
    retcode = -1
    if hasmethod(LuxLib._cublaslt_matmul_fused!,
        (typeof(z), typeof(NNlib.gelu), typeof(weight), typeof(x), typeof(b)))
        y = similar(z)  # break aliasing
        retcode = LuxLib._cublaslt_matmul_fused!(z, NNlib.gelu, weight, x, b, y)
        if retcode == -1
            @warn "cuBLASLt failed for the given inputs $(NNlib.gelu), $(typeof(weight)) \
                [$(size(weight))], $(typeof(x)) [$(size(x))], $(typeof(b)) \
                [$(__length(b))]. Falling back to generic implementation." maxlog=1
        end
    else
        @warn "cuBLASLt not available. Falling back to generic implementation." maxlog=1
    end

    if retcode == -1
        # Generic Fallback: break aliasing in _apply_bias_activation!!
        mul!(z, weight, x)
        z, y = LuxLib.__apply_bias_activation!!(NNlib.gelu, z, b, Val(true))
    end

    ∇__fused_dense_bias_activation_impl_cublaslt = @closure Δ -> begin
        ∂y = LuxLib.__activation_gradient(CRC.unthunk(Δ), z, NNlib.gelu, y)
        ∂b = LuxLib.__added_bias_gradient(b, ∂y)
        ∂x = weight' * ∂y
        ∂w = ∂y * x'
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b
    end

    return z, ∇__fused_dense_bias_activation_impl_cublaslt
end

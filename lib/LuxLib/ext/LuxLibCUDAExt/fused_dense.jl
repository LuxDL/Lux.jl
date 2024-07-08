__length(x) = length(x)
__length(::Nothing) = nothing

function __might_use_cuBLASLt(::Z, ::A, ::W, ::X, ::B) where {Z, A, W, X, B}
    cuBLASLt_functional[] || return false
    return hasmethod(LuxLib._cublaslt_matmul_fused!, (Z, A, W, X, B))
end

@stable default_mode="warn" function LuxLib.__fused_dense_bias_activation_impl(
        act::F, weight::AnyCuMatrix, x::AnyCuMatrix, b::Optional{<:AnyCuVector}) where {F}
    y = similar(x, LuxLib.__get_concrete_fba_output_eltype(act, weight, x, b),
        size(weight, 1), size(x, 2))
    if __might_use_cuBLASLt(y, act, weight, x, b)
        retcode = LuxLib._cublaslt_matmul_fused!(y, act, weight, x, b)
        retcode == 0 && return y
        # cuBLASLt failed for the given inputs use the generic fallback
        @warn "cuBLASLt failed for the given inputs $(act), $(typeof(weight)) \
               [$(size(weight))], $(typeof(x)) [$(size(x))], $(typeof(b)) \
               [$(__length(b))]. Falling back to generic implementation." maxlog=1
    else
        @warn "cuBLASLt not available. Falling back to generic implementation." maxlog=1
    end
    LuxLib.__matmul!(y, weight, x)
    return LuxLib.__apply_bias_activation!!(act, y, b, Val(false))
end

## Special Reverse Pass for gelu activation. All other cases, we don't need special handling
function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(LuxLib.__fused_dense_bias_activation_impl), act::typeof(NNlib.gelu),
        weight::AnyCuMatrix, x::AnyCuMatrix, b::Union{AnyCuVector, Nothing})
    z = similar(x, LuxLib.__get_concrete_fba_output_eltype(NNlib.gelu, weight, x, b),
        size(weight, 1), size(x, 2))
    y = z # aliased for now for type stability
    retcode = -1
    if __might_use_cuBLASLt(z, act, weight, x, b)
        y = similar(z)  # break aliasing
        retcode = LuxLib._cublaslt_matmul_fused!(z, act, weight, x, b, y)
        if retcode == -1
            @warn "cuBLASLt failed for the given inputs $(act), $(typeof(weight)) \
                [$(size(weight))], $(typeof(x)) [$(size(x))], $(typeof(b)) \
                [$(__length(b))]. Falling back to generic implementation." maxlog=1
        end
    else
        @warn "cuBLASLt not available. Falling back to generic implementation." maxlog=1
    end

    if retcode == -1
        # Generic Fallback: break aliasing in _apply_bias_activation!!
        LuxLib.__matmul!(z, weight, x)
        z, y = LuxLib.__apply_bias_activation!!(act, z, b, Val(true))
    end

    ∇__fused_dense_bias_activation_impl_cublaslt = @closure Δ -> begin
        ∂y = LuxLib.__activation_gradient(CRC.unthunk(Δ), z, act, y)
        ∂w, ∂x, ∂b = LuxLib.__matmul_bias_partials(∂y, weight, x, b)
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b
    end

    return z, ∇__fused_dense_bias_activation_impl_cublaslt
end

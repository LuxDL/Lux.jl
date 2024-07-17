__length(x) = length(x)
__length(::Nothing) = nothing

function __try_cublasLt_fused_matmul(act::F, weight::AnyCuMatrix, x::AnyCuMatrix,
        b::Optional{<:AnyCuVector}, ::Val{cache}) where {F, cache}
    z = similar(x, LuxLib.__get_concrete_fba_output_eltype(act, weight, x, b),
        size(weight, 1), size(x, 2))
    y = z # aliased for now for type stability
    if hasmethod(_cublaslt_matmul_fused!,
        (typeof(z), typeof(act), typeof(weight), typeof(x), typeof(b)))
        cache && (y = similar(z)) # break aliasing
        retcode = _cublaslt_matmul_fused!(z, act, weight, x, b, ifelse(cache, y, nothing))
        retcode == 0 && return (z, y, retcode)
        # cuBLASLt failed for the given inputs use the generic fallback
        @warn "cuBLASLt failed for the given inputs $(act), $(typeof(weight)) \
               [$(size(weight))], $(typeof(x)) [$(size(x))], $(typeof(b)) \
               [$(__length(b))]. Falling back to generic implementation." maxlog=1
    else
        @warn "cuBLASLt not available. Falling back to generic implementation." maxlog=1
    end
    return (z, y, -1)
end

@stable default_mode="warn" function LuxLib.__fused_dense_bias_activation_impl(
        act::F, weight::AnyCuMatrix, x::AnyCuMatrix, b::Optional{<:AnyCuVector}) where {F}
    (y, _, retcode) = __try_cublasLt_fused_matmul(act, weight, x, b, Val(false))
    retcode == 0 && return y
    LuxLib.__matmul!(y, weight, x)
    return LuxLib.__bias_activation_impl!!(act, y, b)
end

## Special Reverse Pass for gelu activation. All other cases, we don't need special handling
@stable default_mode="warn" function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(LuxLib.__fused_dense_bias_activation_impl), act::typeof(NNlib.gelu),
        weight::AnyCuMatrix, x::AnyCuMatrix, b::Optional{<:AnyCuVector})
    (z, y, retcode) = __try_cublasLt_fused_matmul(act, weight, x, b, Val(true))
    if retcode == -1
        # Generic Fallback: break aliasing in _apply_bias_activation!!
        LuxLib.__matmul!(z, weight, x)
        z, y = LuxLib.__apply_bias_activation_cached!!(act, z, b)
    end

    proj_w = CRC.ProjectTo(weight)
    proj_x = CRC.ProjectTo(x)
    proj_b = CRC.ProjectTo(b)
    ∇__fused_dense_bias_activation_impl_cublaslt = @closure Δ -> begin
        ∂y = LuxLib.__activation_gradient(CRC.unthunk(Δ), z, act, y)
        ∂w, ∂x, ∂b = LuxLib.__matmul_bias_partials(∂y, weight, x, b)
        return CRC.NoTangent(), CRC.NoTangent(), proj_w(∂w), proj_x(∂x), proj_b(∂b)
    end

    return z, ∇__fused_dense_bias_activation_impl_cublaslt
end

# Wrappers over Base & LinearAlgen implementations to use poly algs if needed
__matmul(A, B) = A * B
__matmul!(C, A, B) = mul!(C, A, B)
__matmuladd(A, B, C) = muladd(A, B, C)
__matmuladd(A, B, ::Nothing) = __matmul(A, B)

# Our main implementations

function __generic_dense_bias_activation(act::F, weight::AbstractMatrix, x::AbstractMatrix,
        bias::Optional{<:AbstractVector}) where {F}
    act === identity && return __matmuladd(weight, x, bias)
    return __generic_bias_activation(act, __matmul(weight, x), bias)
end

# Why are we catching the implementation at this point and not in `bias_act!` like NNlib?
# Turns out NVIDIA has been shipping a bunch of fused kernels for a while now. We use
# fuse all the operations into a single kernel.

function __fused_dense_bias_activation_impl(
        act::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {F}
    return __fused_dense_bias_activation_impl(
        get_device_type((weight, x)), act, weight, x, b)
end

@stable default_mode="warn" function __fused_dense_bias_activation_impl(
        ::Type{T}, act::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {T, F}
    if act === identity
        b === nothing && return (weight * x)
        return __matmuladd(weight, x, b)
    end
    y = similar(weight, __get_concrete_fba_output_eltype(act, weight, x, b),
        size(weight, 1), size(x, 2))
    __matmul!(y, weight, x)
    return __bias_activation_impl!!(act, y, b)
end

function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(__fused_dense_bias_activation_impl),
        ::Type{DT}, act::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {DT, F}
    T = __get_concrete_fba_output_eltype(act, weight, x, b)
    proj_w = CRC.ProjectTo(weight)
    proj_x = CRC.ProjectTo(x)
    proj_b = CRC.ProjectTo(b)

    if __no_intermediate_needed(act, T)
        y = __fused_dense_bias_activation_impl(act, weight, x, b)
        ∇__fused_dense_bias_activation_impl_no_cached = @closure Δ -> begin
            ∂y = __activation_gradient(CRC.unthunk(Δ), y, act, NotaNumber())
            ∂w, ∂x, ∂b = __matmul_bias_partials(∂y, weight, x, b)
            return ∂∅, ∂∅, ∂∅, proj_w(∂w), proj_x(∂x), proj_b(∂b)
        end
        return y, ∇__fused_dense_bias_activation_impl_no_cached
    end

    if __needs_intermediate_but_has_rrule(act, T)
        y = __matmuladd(weight, x, b)
        z = _fast_activation(act, y)
        ∇__fused_dense_bias_activation_impl_cached_crc = @closure Δ -> begin
            ∂y = __activation_gradient(CRC.unthunk(Δ), z, act, y)
            ∂w, ∂x, ∂b = __matmul_bias_partials(∂y, weight, x, b)
            return ∂∅, ∂∅, ∂∅, proj_w(∂w), proj_x(∂x), proj_b(∂b)
        end
        return z, ∇__fused_dense_bias_activation_impl_cached_crc
    end

    y = similar(weight, T, size(weight, 1), size(x, 2))
    __matmul!(y, weight, x)
    z, pb_f = CRC.rrule_via_ad(cfg, __bias_activation_impl, act, y, b)
    ∇__fused_dense_bias_activation_impl_cached = @closure Δ -> begin
        _, _, ∂y, ∂b = pb_f(Δ)
        ∂w, ∂x, _ = __matmul_bias_partials(∂y, ∂b, weight, x, b)
        return ∂∅, ∂∅, ∂∅, proj_w(∂w), proj_x(∂x), proj_b(∂b)
    end
    return z, ∇__fused_dense_bias_activation_impl_cached
end

# Try to use cuBLASLt if available / possible. The function is defined once CUDA.jl is loaded
function __attempt_cublasLt_fused_matmul end

@stable default_mode="warn" function __fused_dense_bias_activation_impl(
        ::Type{<:LuxCUDADevice}, act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    (y, _, retcode) = __attempt_cublasLt_fused_matmul(act, weight, x, b, Val(false))
    retcode == 0 && return y
    __matmul!(y, weight, x)
    return __bias_activation_impl!!(act, y, b)
end

## Special Reverse Pass for gelu activation. All other cases, we don't need special handling
function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode}, ::Type{<:LuxCUDADevice},
        ::typeof(__fused_dense_bias_activation_impl), ::typeof(gelu),
        weight::AbstractMatrix, x::AbstractMatrix, b::Optional{<:AbstractVector})
    (z, y, retcode) = __attempt_cublasLt_fused_matmul(gelu, weight, x, b, Val(false))
    if retcode == -1 # Generic Fallback: break aliasing in _apply_bias_activation!!
        __matmul!(z, weight, x)
        z, y = __apply_bias_activation_cached!!(gelu, z, b)
    end

    proj_w = CRC.ProjectTo(weight)
    proj_x = CRC.ProjectTo(x)
    proj_b = CRC.ProjectTo(b)
    ∇__fused_dense_bias_activation_impl_cublaslt = @closure Δ -> begin
        ∂y = __activation_gradient(CRC.unthunk(Δ), z, gelu, y)
        ∂w, ∂x, ∂b = __matmul_bias_partials(∂y, weight, x, b)
        return ∂∅, ∂∅, ∂∅, proj_w(∂w), proj_x(∂x), proj_b(∂b)
    end

    return z, ∇__fused_dense_bias_activation_impl_cublaslt
end

function __matmul_bias_partials(∂y, weight, x, bias)
    return __matmul_bias_partials(∂y, __added_bias_gradient(bias, ∂y), weight, x, bias)
end
function __matmul_bias_partials(∂y, ∂b, weight, x, bias)
    ∂w = __matmul(∂y, x')
    ∂x = __matmul(weight', ∂y)
    return ∂w, ∂x, ∂b
end

# Wrappers over Base & LinearAlgen implementations to use poly algs if needed
## We define a special __matmul function so that we can define ForwardDiff rules on it without
## type piracy
__matmul(A, B) = A * B
__matmul!(C, A, B) = mul!(C, A, B)
__matmuladd(A, B, C) = muladd(A, B, C)
__matmuladd(A, B, ::Nothing) = __matmul(A, B)

# Our main implementations

function __generic_dense_bias_activation(act::F, weight::AbstractMatrix, x::AbstractMatrix,
        bias::Optional{<:AbstractVector}) where {F}
    act === identity && return __matmuladd(weight, x, bias)
    return __apply_bias_activation(act, __matmul(weight, x), bias)
end

# Why are we catching the implementation at this point and not in `bias_act!` like NNlib?
# Turns out NVIDIA has been shipping a bunch of fused kernels for a while now. We use
# fuse all the operations into a single kernel.

@stable default_mode="warn" function __fused_dense_bias_activation_impl(
        act::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {F}
    if act === identity
        b === nothing && return (weight * x)
        return __matmuladd(weight, x, b)
    end
    y = similar(weight, __get_concrete_fba_output_eltype(act, weight, x, nothing),
        size(weight, 1), size(x, 2))
    __matmul!(y, weight, x)
    return __apply_bias_activation!!(act, y, b, Val(false))
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__fused_dense_bias_activation_impl), act::F, weight::AbstractMatrix,
        x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
    T = __get_concrete_fba_output_eltype(act, weight, x, b)

    # Case I: Activation Function doesn't require caching the intermediate value
    # See https://github.com/FluxML/NNlib.jl/blob/d85402aa39ddc6386d194e0dad88ab2e514ec5ea/src/bias_act.jl#L59-L60
    if act === identity ||
       isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        y = __fused_dense_bias_activation_impl(act, weight, x, b)
        ∇__fused_dense_bias_activation_impl_no_cached = @closure Δ -> begin
            ∂y = act === identity ? CRC.unthunk(Δ) :
                 __activation_gradient(CRC.unthunk(Δ), y, act, NotaNumber())
            ∂w, ∂x, ∂b = __matmul_bias_partials(∂y, weight, x, b)
            return NoTangent(), NoTangent(), ∂w, ∂x, ∂b
        end
        return y, ∇__fused_dense_bias_activation_impl_no_cached
    end

    # Case II: We can't overwrite `y` directly, but we can use the direct ChainRules
    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        y = __matmuladd(weight, x, b)
        z = __fast_broadcast(act, y)
        ∇__fused_dense_bias_activation_impl_cached_crc = @closure Δ -> begin
            ∂y = __activation_gradient(CRC.unthunk(Δ), z, act, y)
            ∂w, ∂x, ∂b = __matmul_bias_partials(∂y, weight, x, b)
            return NoTangent(), NoTangent(), ∂w, ∂x, ∂b
        end
        return z, ∇__fused_dense_bias_activation_impl_cached_crc
    end

    # Case III: Activation Function requires caching the intermediate value
    y = similar(weight, T, size(weight, 1), size(x, 2))
    __matmul!(y, weight, x)
    z, pb_f = CRC.rrule_via_ad(cfg, __apply_bias_activation, act, y, b)
    ∇__fused_dense_bias_activation_impl_cached = @closure Δ -> begin
        _, _, ∂y, ∂b = pb_f(Δ)
        ∂w, ∂x, _ = __matmul_bias_partials(∂y, ∂b, weight, x, b)
        return NoTangent(), NoTangent(), ∂w, ∂x, ∂b
    end
    return z, ∇__fused_dense_bias_activation_impl_cached
end

function __matmul_bias_partials(∂y, weight, x, bias)
    return __matmul_bias_partials(∂y, __added_bias_gradient(bias, ∂y), weight, x, bias)
end
function __matmul_bias_partials(∂y, ∂b, weight, x, bias)
    ∂w = __matmul(∂y, x')
    ∂x = __matmul(weight', ∂y)
    return ∂w, ∂x, ∂b
end

# The cases here are manually split up else Zygote becomes type unstable.
"""
    fused_dense_bias_activation(σ::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Union{Nothing, AbstractVector}) where {F}

Compute `σ.(weight * x .+ b)` with the best possible implementation available. Currently
this implementation attempts to minimize reallocations by reusing the output buffer for
multiple operations.

## Arguments

  - `σ`: Activation function
  - `weight`: Weight matrix
  - `x`: Input matrix
  - `b`: Bias vector (can be `nothing`)

## Notes on implementation

  - Despite the naming, currently only the activation (σ) is fused with the bias addition.
    Currently this is equivalent to using matrix multiply followed by `NNlib.bias_act!`,
    though this function doesn't call those operations.
  - If any of the inputs, don't support setindexing (aka immutable arrays) we fallback to
    the generic non-mutating implementation.
  - For mixed precision inputs, we use the fallback allocating implementation.
  - Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD
    backends or backends that support mutation. Backends like `Tracker` and `ReverseDiff`
    fallback to the generic implementation.
  - For CUDA Arrays, this uses a special fused implementation via cuBLASLt.
"""
@inline function fused_dense_bias_activation(
        σ::F, weight::AbstractMatrix{T}, x::AbstractMatrix{T}, b::Nothing) where {F, T}
    return fused_dense_bias_activation(σ, weight, __is_immutable_array_val(weight), x,
        __is_immutable_array_val(x), b, __is_immutable_array_val(b))
end

@inline function fused_dense_bias_activation(
        σ::F, weight::AbstractMatrix{T}, x::AbstractMatrix{T},
        b::AbstractVector{T}) where {F, T}
    return fused_dense_bias_activation(σ, weight, __is_immutable_array_val(weight), x,
        __is_immutable_array_val(x), b, __is_immutable_array_val(b))
end

@inline function fused_dense_bias_activation(
        σ::F, weight::AbstractMatrix, ::Val{false}, x::AbstractMatrix,
        ::Val{false}, b::Union{Nothing, AbstractVector}, ::Val{false}) where {F}
    return __fused_dense_bias_activation_impl(σ, weight, x, b)
end

@inline function fused_dense_bias_activation(
        σ::F, weight::AbstractMatrix, ::Val, x::AbstractMatrix,
        ::Val, b::Union{Nothing, AbstractVector}, ::Val) where {F}
    return __generic_dense_bias_activation(σ, weight, x, b)
end

# Mixed Precision Casex
@inline function fused_dense_bias_activation(
        σ::F, weight::AbstractMatrix{wT}, x::AbstractMatrix{xT},
        b::AbstractVector{bT}) where {F, wT, xT, bT}
    return __generic_dense_bias_activation(σ, weight, x, b)
end

@inline function fused_dense_bias_activation(σ::F, weight::AbstractMatrix{wT},
        x::AbstractMatrix{xT}, b::Nothing) where {F, wT, xT}
    return __generic_dense_bias_activation(σ, weight, x, b)
end

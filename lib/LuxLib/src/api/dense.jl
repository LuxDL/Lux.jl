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
    We are working towards using faster hardware specific fused kernels for this operation.
    Currently this is equivalent to using matrix multiply followed by `NNlib.bias_act!`,
    though this function doesn't call those operations.
  - If any of the inputs, don't support setindexing (aka immutable arrays) we fallback to
    the generic non-mutating implementation.
  - For mixed precision inputs, we use the fallback allocating implementation.
  - Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD
    backends or backends that support mutation. Backends like `Tracker` and `ReverseDiff`
    fallback to the generic implementation.
"""
@inline function fused_dense_bias_activation(
        σ::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Union{Nothing, AbstractVector}) where {F}
    (__any_immutable_array(weight, x, b) || __is_mixed_precision(weight, x, b)) &&
        return __generic_dense_bias_activation(σ, weight, x, b)
    return __fused_dense_bias_activation_impl(σ, weight, x, b)
end

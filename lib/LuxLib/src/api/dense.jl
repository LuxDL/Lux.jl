# The cases here are manually split up else Zygote becomes type unstable.
"""
    fused_dense_bias_activation(σ::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {F}

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
  - Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD
    backends or backends that support mutation. Backends like `Tracker` and `ReverseDiff`
    fallback to the generic implementation.
  - For CUDA Arrays, this uses a special fused implementation via cuBLASLt.
"""
function fused_dense_bias_activation(σ::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {F}
    return fused_dense_bias_activation(sleefpirates_activation(σ, weight, x, b),
        __is_immutable_array_or_dual_val((weight, x, b)), weight, x, b)
end

for (check, fop) in (
    (false, :__fused_dense_bias_activation_impl), (true, :__generic_dense_bias_activation))
    @eval function fused_dense_bias_activation(
            σ::F, ::Val{$(check)}, weight::AbstractMatrix,
            x::AbstractMatrix, b::Optional{<:AbstractVector}) where {F}
        return $(fop)(σ, weight, x, b)
    end
end

"""
    fused_conv_bias_activation(σ::F, weight::AbstractArray, x::AbstractArray,
        b::Optional{<:AbstractVector}, cdims::ConvDims) where {F}

Computes `σ.(conv(x, weight, cdims) .+ b)` (`b` is not exactly broadcasted like this,
rather it is reshaped and broadcasted to the penultimate dimension) with the best possible
implementation available. This operation fuses operations into a single kernel if possible,
and minimizes reallocations by reusing the output buffer for multiple operations.

## Arguments

  - `σ`: Activation function
  - `weight`: Weight tensor
  - `x`: Input tensor
  - `b`: Bias tensor (can be `nothing`)
  - `cdims`: `ConvDims` object

## Notes on implementation

  - For CUDA Arrays, this uses fused CUDNN kernels when the activation is `identity` or
    `relu`. For other activations, it tries to fuse the operations on the Julia side.
  - If any of the inputs, don't support setindexing (aka immutable arrays) we fallback to
    the generic non-mutating implementation.
  - Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD
    backends or backends that support mutation. Backends like `Tracker` and `ReverseDiff`
    fallback to the generic implementation.
  - For Mixed-Precision Inputs on GPU, we type promote the inputs to the highest precision,
    with a warning.
"""
function fused_conv_bias_activation(
        σ::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        b::Optional{<:AbstractVector}, cdims::ConvDims) where {F, N, wT, xT}
    σ′ = get_impl(:select_fastest_activation)(σ, weight, x, b)
    return get_impl(:fused_conv)(σ′, weight, x, b, cdims)
end

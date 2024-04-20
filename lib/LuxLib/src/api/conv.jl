"""
    fused_conv_bias_activation(σ::F, weight::AbstractArray, x::AbstractArray,
        b::Union{Nothing, AbstractArray}, cdims::ConvDims) where {F}

Computes `σ.(conv(x, weight, cdims) .+ b)` with the best possible implementation available.
This operation fuses operations into a single kernel if possible, and minimizes
reallocations by reusing the output buffer for multiple operations.

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
  - For mixed precision inputs, we use the fallback allocating implementation.
  - Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD
    backends or backends that support mutation. Backends like `Tracker` and `ReverseDiff`
    fallback to the generic implementation.
"""
@inline function fused_conv_bias_activation(σ::F, weight::AbstractArray, x::AbstractArray,
        b::Union{Nothing, AbstractArray}, cdims::ConvDims) where {F}
    b !== nothing && @assert ndims(b) == ndims(weight) == ndims(x)
    (__any_immutable_array(weight, x, b) || __is_mixed_precision(weight, x, b)) &&
        return __generic_conv_bias_activation(σ, weight, x, b, cdims)
    return __fused_conv_bias_activation_impl(σ, weight, x, b, cdims)
end

# copy a subarray to make it contiguous in memory
@inline function fused_conv_bias_activation(σ::F, weight::AbstractArray, x::SubArray,
        b::Union{Nothing, AbstractArray}, cdims::ConvDims) where {F}
    b !== nothing && @assert ndims(b) == ndims(weight) == ndims(x)
    return fused_conv_bias_activation(σ, weight, copy(x), b, cdims)
end

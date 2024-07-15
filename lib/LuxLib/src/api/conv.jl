# The cases here are manually split up else Zygote becomes type unstable.
"""
    fused_conv_bias_activation(σ::F, weight::AbstractArray, x::AbstractArray,
        b::Optional{<:AbstractArray}, cdims::ConvDims) where {F}

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
  - Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD
    backends or backends that support mutation. Backends like `Tracker` and `ReverseDiff`
    fallback to the generic implementation.
  - For Mixed-Precision Inputs on GPU, we type promote the inputs to the highest precision,
    with a warning.
"""
@stable default_mode="warn" function fused_conv_bias_activation(args...)
    return _fused_conv_bias_activation(args...)
end

# Needed for Zygote type-stability
function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(fused_conv_bias_activation), args...)
    return CRC.rrule_via_ad(cfg, _fused_conv_bias_activation, args...)
end

function _fused_conv_bias_activation(
        σ::F, weight::AbstractArray{<:Number, N}, x::AbstractArray{<:Number, N},
        b::Optional{<:AbstractArray{<:Number, N}}, cdims::ConvDims) where {F, N}
    return _fused_conv_bias_activation(
        σ, __is_immutable_array_or_dual_val((weight, x, b)), weight, x, b, cdims)
end

for (check, fop) in (
    (false, :_fused_conv_bias_activation_impl), (true, :_generic_conv_bias_activation))
    @eval function _fused_conv_bias_activation(
            σ::F, ::Val{$(check)}, weight::AbstractArray{<:Number, N},
            x::AbstractArray{<:Number, N},
            b::Optional{<:AbstractArray}, cdims::ConvDims) where {F, N}
        return $(fop)(σ, weight, x, b, cdims)
    end
end

# The cases here are manually split up else Zygote becomes type unstable.
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
  - For Mixed-Precision Inputs on GPU, we type promote the inputs to the highest precision,
    with a warning.
"""
function fused_conv_bias_activation end

# Avoid Ambiguity
for aType in (AbstractArray, GPUArraysCore.AnyGPUArray)
    @eval begin
        @inline function fused_conv_bias_activation(
                σ::F, weight::$(aType){T, N}, x::$(aType){T, N},
                b::$(aType){T, N}, cdims::ConvDims) where {F, T, N}
            return fused_conv_bias_activation(
                σ, weight, __is_immutable_array_val(weight), x,
                __is_immutable_array_val(x), b, __is_immutable_array_val(b), cdims)
        end

        @inline function fused_conv_bias_activation(
                σ::F, weight::$(aType){T, N}, x::$(aType){T, N},
                b::Nothing, cdims::ConvDims) where {F, T, N}
            return fused_conv_bias_activation(
                σ, weight, __is_immutable_array_val(weight), x,
                __is_immutable_array_val(x), b, __is_immutable_array_val(b), cdims)
        end
    end
end

@inline function fused_conv_bias_activation(
        σ::F, weight::AbstractArray, ::Val{false}, x::AbstractArray, ::Val{false},
        b::Union{Nothing, AbstractArray}, ::Val{false}, cdims::ConvDims) where {F}
    return __fused_conv_bias_activation_impl(σ, weight, x, b, cdims)
end

@inline function fused_conv_bias_activation(
        σ::F, weight::AbstractArray, ::Val, x::AbstractArray, ::Val,
        b::Union{Nothing, AbstractArray}, ::Val, cdims::ConvDims) where {F}
    return __generic_conv_bias_activation(σ, weight, x, b, cdims)
end

# SubArray Inputs: copy a subarray to make it contiguous in memory
@inline function fused_conv_bias_activation(
        σ::F, weight::AbstractArray{wT, N}, x::SubArray{xT, N},
        b::AbstractArray{bT, N}, cdims::ConvDims) where {F, wT, xT, bT, N}
    return fused_conv_bias_activation(σ, weight, copy(x), b, cdims)
end

@inline function fused_conv_bias_activation(
        σ::F, weight::AbstractArray{wT, N}, x::SubArray{xT, N},
        b::Nothing, cdims::ConvDims) where {F, wT, xT, N}
    return fused_conv_bias_activation(σ, weight, copy(x), b, cdims)
end

# Mixed Precision Generic (Non GPU) Inputs: Code in NNlib can handle this case, but not for
# the GPU case
@inline function fused_conv_bias_activation(
        σ::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        b::AbstractArray{bT, N}, cdims::ConvDims) where {F, wT, xT, bT, N}
    return __generic_conv_bias_activation(σ, weight, x, b, cdims)
end

@inline function fused_conv_bias_activation(
        σ::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        b::Nothing, cdims::ConvDims) where {F, wT, xT, N}
    return __generic_conv_bias_activation(σ, weight, x, b, cdims)
end

# Mixed Precision GPU Inputs
@inline function fused_conv_bias_activation(
        σ::F, weight::GPUArraysCore.AnyGPUArray{wT, N}, x::GPUArraysCore.AnyGPUArray{xT, N},
        b::GPUArraysCore.AnyGPUArray{bT, N}, cdims::ConvDims) where {F, wT, xT, bT, N}
    T = __get_concrete_fba_output_eltype(σ, weight, x, b)
    @warn "Mixed Precision Inputs on GPU for `fused_conv_bias_activation`. Promoting \
           computation to $T" weight=wT x=xT bias=bT maxlog=1
    return fused_conv_bias_activation(
        σ, _oftype_array(T, weight), _oftype_array(T, x), _oftype_array(T, b), cdims)
end

@inline function fused_conv_bias_activation(
        σ::F, weight::GPUArraysCore.AnyGPUArray{wT, N}, x::GPUArraysCore.AnyGPUArray{xT, N},
        b::Nothing, cdims::ConvDims) where {F, wT, xT, N}
    T = __get_concrete_fba_output_eltype(σ, weight, x, b)
    @warn "Mixed Precision Inputs on GPU for `fused_conv_bias_activation`. Promoting \
           computation to $T" weight=wT x=xT maxlog=1
    return fused_conv_bias_activation(
        σ, _oftype_array(T, weight), _oftype_array(T, x), b, cdims)
end

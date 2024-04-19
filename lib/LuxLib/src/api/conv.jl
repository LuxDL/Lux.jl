@inline function fused_conv_bias_activation(σ::F, weight::AbstractArray, x::AbstractArray,
        b::Union{Nothing, AbstractArray}, cdims::ConvDims) where {F}
    b !== nothing && @assert ndims(b) == ndims(weight) == ndims(x)
    (__any_immutable_array(weight, x, b) || __is_mixed_precision(weight, x, b)) &&
        return __generic_conv_bias_activation(σ, weight, x, b, cdims)
    return __fused_conv_bias_activation_impl(σ, weight, x, b, cdims)
end

# For Dense GPU Arrays we have faster implementations, so make the copy!
@inline function fused_conv_bias_activation(
        σ::F, weight::AbstractArray, x::SubArray{xT, N, <:AnyGPUArray},
        b::Union{Nothing, AbstractArray}, cdims::ConvDims) where {xT, N, F}
    b !== nothing && @assert ndims(b) == ndims(weight) == ndims(x)
    return fused_conv_bias_activation(σ, weight, copy(x), b, cdims)
end

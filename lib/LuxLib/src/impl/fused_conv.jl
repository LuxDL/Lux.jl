@inline function __generic_conv_bias_activation(
        act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Union{Nothing, AbstractArray}, cdims::ConvDims) where {wT, xT, N, F}
    return __apply_bias_activation(act, conv(x, weight, cdims), bias)
end

# This implementation is different from `conv_bias_act` in that it defines the proper rrules
# and fuses operations into a single kernel if it is possible. Unfortinately there are
# certain configurations where CUDNN allows caching intermediates, but we don't do that rn.

@inline function __fused_conv_bias_activation_impl(
        act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Union{Nothing, AbstractArray}, cdims::ConvDims) where {wT, xT, N, F}
    if act === identity
        bias === nothing && return conv(x, weight, cdims)
        return NNlib.conv_bias_act(x, weight, cdims, bias, identity)
    end
    # cuDNN has a fused kernel only for relu
    act === relu && return NNlib.conv_bias_act(x, weight, cdims, bias, act)
    # just fusing bias doesn't make sense when we can fuse them both on the julia side
    y = similar(x, __get_concrete_fba_output_eltype(act, weight, x, bias),
        NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, N))
    conv!(y, x, weight, cdims)
    return __apply_bias_activation!!(act, y, bias, Val(false))
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__fused_conv_bias_activation_impl),
        act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Union{Nothing, AbstractArray}, cdims::ConvDims) where {wT, xT, N, F}
    T = __get_concrete_fba_output_eltype(act, weight, x, bias)
    y = similar(x, T, NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, N))

    # Will be true for identity and relu as well but still to be certain
    if act === relu ||
       act === identity ||
       isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        if act === relu || act === identity
            NNlib.conv_bias_act!(y, x, weight, cdims, bias, act)
        else
            conv!(y, x, weight, cdims)
            y = __apply_bias_activation!!(act, y, bias, Val(false))
        end
        ∇__fused_conv_bias_activation_impl_no_cached = @closure Δ -> begin
            ∂y = act === identity ? CRC.unthunk(Δ) :
                 only_derivative.(y, act, NotaNumber()) .* CRC.unthunk(Δ)
            ∂b = __added_bias_gradient(bias, ∂y)
            ∂x = NNlib.∇conv_data(∂y, weight, cdims)
            ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
            return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b, CRC.NoTangent()
        end
        return y, ∇__fused_conv_bias_activation_impl_no_cached
    end

    # In any case here we need the intermediate pre-activation values
    conv!(y, x, weight, cdims)

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        z, y = __apply_bias_activation!!(act, y, bias, Val(true))
        ∇__fused_conv_bias_activation_impl_cached_crc = @closure Δ -> begin
            ∂y = only_derivative.(z, act, y) .* CRC.unthunk(Δ)
            ∂b = __added_bias_gradient(bias, ∂y)
            ∂x = NNlib.∇conv_data(∂y, weight, cdims)
            ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
            return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b, CRC.NoTangent()
        end
        return z, ∇__fused_conv_bias_activation_impl_cached_crc
    end

    z, pb_f = CRC.rrule_via_ad(cfg, __apply_bias_activation, act, y, bias)
    ∇__fused_conv_bias_activation_impl_cached = @closure Δ -> begin
        _, ∂y, ∂b = pb_f(Δ)
        ∂x = NNlib.∇conv_data(∂y, weight, cdims)
        ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b, CRC.NoTangent()
    end

    return z, ∇__fused_conv_bias_activation_impl_cached
end

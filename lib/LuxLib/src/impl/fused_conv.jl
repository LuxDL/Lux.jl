@inline function __generic_conv_bias_activation(
        ::typeof(identity), weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Union{Nothing, AbstractArray}, cdims::ConvDims) where {wT, xT, N}
    y = conv(x, weight, cdims)
    bias === nothing && return y
    return y .+ bias
end

@inline function __generic_conv_bias_activation(
        act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Union{Nothing, AbstractArray}, cdims::ConvDims) where {wT, xT, N, F}
    y = conv(x, weight, cdims)
    bias === nothing && return act.(y)
    return act.(y .+ bias)
end

# This implementation is different from `conv_bias_act` in that it defines the proper rrules
# and fuses operations into a single kernel if it is possible. Unfortinately there are
# certain configurations where CUDNN allows caching intermediates, but we don't do that rn.

@inline function __fused_conv_bias_activation_impl(
        ::typeof(identity), weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Nothing, cdims::ConvDims) where {wT, xT, N}
    return conv(x, weight, cdims)
end

@inline function __fused_conv_bias_activation_impl(
        ::typeof(identity), weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::AbstractArray{bT, N}, cdims::ConvDims) where {wT, xT, bT, N}
    return NNlib.conv_bias_act(x, weight, cdims, bias, identity)
end

function CRC.rrule(::typeof(__fused_conv_bias_activation_impl), ::typeof(identity),
        weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::AbstractArray{bT, N}, cdims::ConvDims) where {wT, xT, bT, N}
    y = __fused_conv_bias_activation_impl(identity, weight, x, bias, cdims)
    ∇__fused_conv_bias_activation_impl = @closure Δ -> begin
        ∂y = CRC.unthunk(Δ)
        ∂b = similar(bias)
        sum!(∂b, ∂y)
        ∂x = NNlib.∇conv_data(∂y, weight, cdims)
        ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b, CRC.NoTangent()
    end
    return y, ∇__fused_conv_bias_activation_impl
end

@inline function __fused_conv_bias_activation_impl(
        act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Union{Nothing, AbstractArray}, cdims::ConvDims) where {wT, xT, N, F}
    # cuDNN has a fused kernel only for relu
    act === relu && return NNlib.conv_bias_act(x, weight, cdims, bias, act)
    # just fusing bias doesn't make sense when we can fuse them both on the julia side
    y = similar(x, __get_concrete_fba_output_eltype(act, weight, x, bias),
        NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, N))
    conv!(y, x, weight, cdims)
    if bias === nothing
        @. y = act(y)
    else
        @. y = act(y + bias)
    end
    return y
end

function CRC.rrule(
        ::typeof(__fused_conv_bias_activation_impl), act::F, weight::AbstractArray{wT, N},
        x::AbstractArray{xT, N}, bias::Nothing, cdims::ConvDims) where {wT, xT, N, F}
    T = __get_concrete_fba_output_eltype(act, weight, x, bias)
    y = similar(x, T, NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, N))
    conv!(y, x, weight, cdims)

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        @. y = act(y)
        ∇__fused_conv_bias_activation_impl_no_cached = @closure Δ -> begin
            ∂y = only_derivative.(y, act, NotaNumber()) .* CRC.unthunk(Δ)
            ∂x = NNlib.∇conv_data(∂y, weight, cdims)
            ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
            return (
                CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, CRC.NoTangent(), CRC.NoTangent())
        end
        return y, ∇__fused_conv_bias_activation_impl_no_cached
    end

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        z = @. act(y)
        ∇__fused_conv_bias_activation_impl_cached_crc = @closure Δ -> begin
            ∂y = only_derivative.(z, act, y) .* CRC.unthunk(Δ)
            ∂x = NNlib.∇conv_data(∂y, weight, cdims)
            ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
            return (
                CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, CRC.NoTangent(), CRC.NoTangent())
        end
        return y, ∇__fused_conv_bias_activation_impl_cached_crc
    end

    z, pb_f = CRC.rrule_via_ad(cfg, Base.Fix1(broadcast, act), y)
    ∇__fused_conv_bias_activation_impl_cached = @closure Δ -> begin
        _, ∂y = pb_f(Δ)
        ∂x = NNlib.∇conv_data(∂y, weight, cdims)
        ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, CRC.NoTangent(), CRC.NoTangent()
    end

    return z, ∇__fused_conv_bias_activation_impl_cached
end

function CRC.rrule(::typeof(__fused_conv_bias_activation_impl), act::F,
        weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::AbstractArray{bT, N}, cdims::ConvDims) where {wT, xT, bT, N, F}
    T = __get_concrete_fba_output_eltype(act, weight, x, bias)
    y = similar(x, T, NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, N))

    if act === relu ||
       isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        if act === relu
            NNlib.conv_bias_act!(y, x, weight, cdims, bias, act)
        else
            conv!(y, x, weight, cdims)
            @. y = act(y + bias)
        end

        ∇__fused_conv_bias_activation_impl_no_cached = @closure Δ -> begin
            ∂y = only_derivative.(y, act, NotaNumber()) .* CRC.unthunk(Δ)
            ∂b = similar(bias)
            sum!(∂b, ∂y)
            ∂x = NNlib.∇conv_data(∂y, weight, cdims)
            ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
            return (CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b, CRC.NoTangent())
        end
        return y, ∇__fused_conv_bias_activation_impl_no_cached
    end

    conv!(y, x, weight, cdims)

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        @. y += bias
        z = @. act(y)
        ∇__fused_conv_bias_activation_impl_cached_crc = @closure Δ -> begin
            ∂y = only_derivative.(z, act, y) .* CRC.unthunk(Δ)
            ∂b = similar(bias)
            sum!(∂b, ∂y)
            ∂x = NNlib.∇conv_data(∂y, weight, cdims)
            ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
            return (CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b, CRC.NoTangent())
        end
        return z, ∇__fused_conv_bias_activation_impl_cached_crc
    end

    z, pb_f = CRC.rrule_via_ad(cfg, @closure((y, b)->@.(act(y + b))), y, bias)
    ∇__fused_conv_bias_activation_impl_cached = @closure Δ -> begin
        _, ∂y, ∂b = pb_f(Δ)
        ∂x = NNlib.∇conv_data(∂y, weight, cdims)
        ∂w = NNlib.∇conv_filter(x, ∂y, cdims)
        return CRC.NoTangent(), CRC.NoTangent(), ∂w, ∂x, ∂b, CRC.NoTangent()
    end

    return z, ∇__fused_conv_bias_activation_impl_cached
end

# wrappers over NNlib implementations to handle mixed precision inputs
@inline function __gpu_get_weight_input(::Type{wT}, ::Type{xT}, weight, x) where {wT, xT}
    T = promote_type(xT, wT)
    @warn "Mixed Precision Inputs received for GPU convolution [weight: $(wT) and x: \
           $(xT)]. Promoting to $(wT)." maxlog=1
    return (__materialize_subarray(_oftype_array(T, weight)),
        __materialize_subarray(_oftype_array(T, x)))
end
@inline function __gpu_get_weight_input(::Type{T}, ::Type{T}, weight, x) where {T}
    return __materialize_subarray(weight), __materialize_subarray(x)
end

@inline __depthwiseconv(x, weight, cdims) = NNlib.depthwiseconv(x, weight, cdims)

@inline __conv!(y, x, weight, cdims) = conv!(
    y, __materialize_subarray(x), __materialize_subarray(weight), cdims)
@inline function __conv!(y::AnyGPUArray{yT, N}, x::AnyGPUArray{xT, N},
        weight::AnyGPUArray{wT, N}, cdims) where {yT, xT, wT, N}
    if xT !== wT !== yT
        @warn "Mixed Precision Inputs received for GPU convolution [weight: $(wT) and x: \
               $(xT)]. Promoting to $(yT)." maxlog=1
    end
    return conv!(y, __materialize_subarray(_oftype_array(yT, x)),
        __materialize_subarray(_oftype_array(yT, weight)), cdims)
end

@inline __conv(x, weight, cdims) = conv(
    __materialize_subarray(x), __materialize_subarray(weight), cdims)
@inline function __conv(
        x_::AnyGPUArray{xT, N}, weight_::AnyGPUArray{wT, N}, cdims) where {xT, wT, N}
    weight, x = __gpu_get_weight_input(wT, xT, weight_, x_)
    T = promote_type(eltype(x), eltype(weight))
    if eltype(x) !== eltype(weight)
        @warn "Mixed Precision Inputs received for GPU convolution [weight: $(eltype(weight)) and x: \
               $(eltype(x))]. Promoting to $(eltype(x))." maxlog=1
    end
    return conv(__materialize_subarray(_oftype_array(T, x)),
        __materialize_subarray(_oftype_array(T, weight)), cdims)
end

@inline __∇conv_data(x, weight, cdims) = ∇conv_data(
    __materialize_subarray(x), __materialize_subarray(weight), cdims)
@inline function __∇conv_data(
        x_::AnyGPUArray{xT, N}, weight_::AnyGPUArray{wT, N}, cdims) where {xT, wT, N}
    weight, x = __gpu_get_weight_input(wT, xT, weight_, x_)
    T = promote_type(eltype(x), eltype(weight))
    if eltype(x) !== eltype(weight)
        @warn "Mixed Precision Inputs received for GPU convolution [weight: $(eltype(weight)) and x: \
               $(eltype(x))]. Promoting to $(eltype(x))." maxlog=1
    end
    return ∇conv_data(__materialize_subarray(_oftype_array(T, x)),
        __materialize_subarray(_oftype_array(T, weight)), cdims)
end

@inline __∇conv_filter(x, y, cdims) = ∇conv_filter(
    __materialize_subarray(x), __materialize_subarray(y), cdims)
@inline function __∇conv_filter(
        x_::AnyGPUArray{xT, N}, y_::AnyGPUArray{yT, N}, cdims) where {xT, yT, N}
    y, x = __gpu_get_weight_input(yT, xT, y_, x_)
    T = promote_type(eltype(x), eltype(y))
    if eltype(x) !== eltype(y)
        @warn "Mixed Precision Inputs received for GPU convolution [weight: $(eltype(y)) and x: \
               $(eltype(x))]. Promoting to $(eltype(x))." maxlog=1
    end
    return ∇conv_filter(__materialize_subarray(_oftype_array(T, x)),
        __materialize_subarray(_oftype_array(T, y)), cdims)
end

@inline __conv_bias_act(x, weight, cdims, bias, act::F) where {F} = __conv_bias_act_impl(
    __materialize_subarray(x), __materialize_subarray(weight), cdims, bias, act)
@inline function __conv_bias_act(x_::AnyGPUArray{xT, N}, weight_::AnyGPUArray{wT, N},
        cdims, bias, act::F) where {xT, wT, N, F}
    weight, x = __gpu_get_weight_input(wT, xT, weight_, x_)
    bias !== nothing && (bias = _oftype_array(eltype(x), bias))
    return __conv_bias_act_impl(x, weight, cdims, bias, act)
end

@inline function __conv_bias_act_impl(x, weight, cdims, bias, act::F) where {F}
    y = similar(x, __get_concrete_fba_output_eltype(act, weight, x, bias),
        NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, ndims(x)))
    __conv!(y, x, weight, cdims)
    return __apply_bias_activation!!(act, y, bias, Val(false))
end
@inline function __conv_bias_act_impl(x::AnyGPUArray, weight, cdims, bias, act::F) where {F}
    bias === nothing && return fast_activation!!(act, __conv(x, weight, cdims))
    if act === identity || act === relu
        return NNlib.conv_bias_act(x, weight, cdims, bias, act)
    end
    y = similar(x, __get_concrete_fba_output_eltype(act, weight, x, bias),
        NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, ndims(x)))
    __conv!(y, x, weight, cdims)
    return __apply_bias_activation!!(act, y, bias, Val(false))
end

# Our main implementations
@inline function _generic_conv_bias_activation(
        act::F, weight::AbstractArray, args...) where {F}
    old_threads = __maybe_reduce_BLAS_threads(weight)
    ret = __generic_conv_bias_activation(act, weight, args...)
    __reset_BLAS_threads(old_threads)
    return ret
end

@inline function __generic_conv_bias_activation(
        act::F, weight::AbstractArray{<:Number, N}, x::AbstractArray{<:Number, N},
        bias::Optional{<:AbstractArray}, cdims::ConvDims) where {F, N}
    return __apply_bias_activation(act, __conv(x, weight, cdims), bias)
end

# This implementation is different from `conv_bias_act` in that it defines the proper rrules
# and fuses operations into a single kernel if it is possible. Unfortunately there are
# certain configurations where CUDNN allows caching intermediates, but we don't do that rn.

@inline function _fused_conv_bias_activation_impl(
        act::F, weight::AbstractArray, args...) where {F}
    old_threads = __maybe_reduce_BLAS_threads(weight)
    ret = __fused_conv_bias_activation_impl(act, weight, args...)
    __reset_BLAS_threads(old_threads)
    return ret
end

@stable default_mode="warn" function __fused_conv_bias_activation_impl(
        act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Optional{<:AbstractArray}, cdims::ConvDims) where {wT, xT, N, F}
    return __conv_bias_act(x, weight, cdims, bias, act)
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__fused_conv_bias_activation_impl),
        act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Optional{<:AbstractArray}, cdims::ConvDims) where {wT, xT, N, F}
    T = __get_concrete_fba_output_eltype(act, weight, x, bias)

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        y = __conv_bias_act(x, weight, cdims, bias, act)
        ∇__fused_conv_bias_activation_impl_no_cached = @closure Δ -> begin
            old_threads = __maybe_reduce_BLAS_threads(weight)
            Δ = CRC.unthunk(NNlib.colmajor(Δ))
            ∂y = act === identity ? Δ : __activation_gradient(Δ, y, act, NotaNumber())
            ∂w, ∂x, ∂b = __conv_bias_partials(∂y, weight, x, bias, cdims)
            __reset_BLAS_threads(old_threads)
            return NoTangent(), NoTangent(), ∂w, ∂x, ∂b, NoTangent()
        end
        return y, ∇__fused_conv_bias_activation_impl_no_cached
    end

    # In any case here we need the intermediate pre-activation values
    y = similar(x, T, NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, N))
    __conv!(y, x, weight, cdims)

    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        z, y = __apply_bias_activation!!(act, y, bias, Val(true))
        ∇__fused_conv_bias_activation_impl_cached_crc = @closure Δ -> begin
            old_threads = __maybe_reduce_BLAS_threads(weight)
            Δ = CRC.unthunk(NNlib.colmajor(Δ))
            ∂y = __activation_gradient(Δ, z, act, y)
            ∂w, ∂x, ∂b = __conv_bias_partials(∂y, weight, x, bias, cdims)
            __reset_BLAS_threads(old_threads)
            return NoTangent(), NoTangent(), ∂w, ∂x, ∂b, NoTangent()
        end
        return z, ∇__fused_conv_bias_activation_impl_cached_crc
    end

    z, pb_f = CRC.rrule_via_ad(cfg, __apply_bias_activation, act, y, bias)
    ∇__fused_conv_bias_activation_impl_cached = @closure Δ -> begin
        old_threads = __maybe_reduce_BLAS_threads(weight)
        Δ = NNlib.colmajor(Δ)
        _, _, ∂y, ∂b = pb_f(Δ)
        ∂w, ∂x, _ = __conv_bias_partials(∂y, ∂b, weight, x, bias, cdims)
        __reset_BLAS_threads(old_threads)
        return NoTangent(), NoTangent(), ∂w, ∂x, ∂b, NoTangent()
    end

    return z, ∇__fused_conv_bias_activation_impl_cached
end

@inline function __conv_bias_partials(∂y, weight, x, bias, cdims)
    return __conv_bias_partials(∂y, __added_bias_gradient(bias, ∂y), weight, x, bias, cdims)
end
@inline function __conv_bias_partials(∂y, ∂b, weight, x, bias, cdims)
    ∂x = __∇conv_data(∂y, weight, cdims)
    ∂w = __∇conv_filter(x, ∂y, cdims)
    return ∂w, ∂x, ∂b
end

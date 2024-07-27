# wrappers over NNlib implementations to handle mixed precision inputs
function __get_conv_input_weight(
        ::Type{<:AbstractGPUDevice}, ::Type{xT}, ::Type{wT}, x, weight) where {xT, wT}
    T = promote_type(xT, wT)
    @warn "Mixed Precision Inputs received for GPU convolution [weight: $(wT)] and \
           [x: $(xT)]. Promoting to $(T)." maxlog=1
    return (__materialize_subarray(_ofeltype_array(T, x)),
        __materialize_subarray(_ofeltype_array(T, weight)))
end
function __get_conv_input_weight(
        ::Type{<:AbstractGPUDevice}, ::Type{T}, ::Type{T}, x, weight) where {T}
    return __materialize_subarray(x), __materialize_subarray(weight)
end
function __get_conv_input_weight(::Type{<:AbstractGPUDevice}, ::Type{<:ForwardDiff.Dual},
        ::Type{T}, x, weight) where {T}
    return __materialize_subarray(x), __materialize_subarray(weight)
end
function __get_conv_input_weight(::Type{<:AbstractGPUDevice}, ::Type{T},
        ::Type{<:ForwardDiff.Dual}, x, weight) where {T}
    return __materialize_subarray(x), __materialize_subarray(weight)
end
function __get_conv_input_weight(::Type{<:AbstractGPUDevice}, ::Type{<:ForwardDiff.Dual},
        ::Type{<:ForwardDiff.Dual}, x, weight)
    return __materialize_subarray(x), __materialize_subarray(weight)
end

function __get_conv_input_weight(
        ::Type{<:AbstractDevice}, ::Type{xT}, ::Type{wT}, x, weight) where {xT, wT}
    return __materialize_subarray(x), __materialize_subarray(weight)
end

__depthwiseconv(x, weight, cdims) = NNlib.depthwiseconv(x, weight, cdims)

__conv!(y, x, weight, cdims) = __conv!(get_device_type((y, x, weight)), y, x, weight, cdims)
function __conv!(::Type{<:AbstractDevice}, y::AbstractArray{<:Number, N},
        x::AbstractArray{<:Number, N},
        weight::AbstractArray{<:Number, N}, cdims::ConvDims) where {N}
    return conv!(y, __materialize_subarray(x), __materialize_subarray(weight), cdims)
end
function __conv!(::Type{<:AbstractGPUDevice}, y::AbstractArray{yT, N},
        x::AbstractArray{xT, N}, weight::AbstractArray{wT, N},
        cdims::ConvDims) where {yT <: Number, xT <: Number, wT <: Number, N}
    if xT !== wT !== yT
        @warn "Mixed Precision Inputs received for GPU convolution [weight: $(wT)] and \
               [x: $(xT)]. Promoting to $(yT)." maxlog=1
    end
    return conv!(y, __materialize_subarray(_ofeltype_array(yT, x)),
        __materialize_subarray(_ofeltype_array(yT, weight)), cdims)
end

function __conv(
        x_::AbstractArray{xT}, weight_::AbstractArray{wT}, cdims::ConvDims) where {xT, wT}
    x, weight = __get_conv_input_weight(get_device_type((x_, weight_)), xT, wT, x_, weight_)
    return conv(x, weight, cdims)
end

function __∇conv_data(
        x_::AbstractArray{xT}, weight_::AbstractArray{wT}, cdims::ConvDims) where {xT, wT}
    x, weight = __get_conv_input_weight(get_device_type((x_, weight_)), xT, wT, x_, weight_)
    return ∇conv_data(x, weight, cdims)
end

function __∇conv_filter(
        x_::AbstractArray{xT}, y_::AbstractArray{yT}, cdims::ConvDims) where {xT, yT}
    x, y = __get_conv_input_weight(get_device_type((x_, y_)), xT, yT, x_, y_)
    return ∇conv_filter(x, y, cdims)
end

function __conv_bias_act(x_::AbstractArray{xT}, weight_::AbstractArray{wT}, cdims::ConvDims,
        bias_::Optional{<:AbstractVector}, act::F) where {xT, wT, F}
    dev = get_device_type((x_, weight_, bias_))
    x, weight = __get_conv_input_weight(dev, xT, wT, x_, weight_)
    bias = _ofeltype_array(eltype(x), bias_)
    return __conv_bias_act_impl(dev, x, weight, cdims, bias, act)
end

function __conv_bias_act_impl(::Type, x, weight, cdims, bias, act::F) where {F}
    y = similar(x, __get_concrete_fba_output_eltype(act, weight, x, bias),
        NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, ndims(x)))
    __conv!(y, x, weight, cdims)
    return __bias_activation_impl!!(act, y, bias)
end
function __conv_bias_act_impl(
        ::Type{<:CUDADevice}, x, weight, cdims, bias, act::F) where {F}
    bias === nothing && return fast_activation!!(act, __conv(x, weight, cdims))
    if act === identity || act === relu
        bias_ = __reshape_bias_into_xdims(x, bias)
        return NNlib.conv_bias_act(x, weight, cdims, bias_, act)
    end
    return __conv_bias_act_impl(Nothing, x, weight, cdims, bias, act)
end

# Our main implementations
function _generic_conv_bias_activation(
        act::F, weight::AbstractArray{<:Number, N}, x::AbstractArray{<:Number, N},
        bias::Optional{<:AbstractVector}, cdims::ConvDims) where {F, N}
    old_threads = __maybe_reduce_BLAS_threads(weight)
    ret = __generic_conv_bias_activation(
        get_device_type((weight, x)), act, weight, x, bias, cdims)
    __reset_BLAS_threads(old_threads)
    return ret
end

function __generic_conv_bias_activation(
        ::Type{T}, act::F, weight::AbstractArray{<:Number, N},
        x::AbstractArray{<:Number, N}, bias::Optional{<:AbstractVector},
        cdims::ConvDims) where {T, F, N}
    return __generic_bias_activation(act, __conv(x, weight, cdims), bias)
end

# This implementation is different from `conv_bias_act` in that it defines the proper rrules
# and fuses operations into a single kernel if it is possible. Unfortunately there are
# certain configurations where CUDNN allows caching intermediates, but we don't do that rn.

function _fused_conv_bias_activation_impl(
        act::F, weight::AbstractArray{<:Number, N}, x::AbstractArray{<:Number, N},
        bias::Optional{<:AbstractVector}, cdims::ConvDims) where {F, N}
    old_threads = __maybe_reduce_BLAS_threads(weight)
    ret = __fused_conv_bias_activation_impl(
        get_device_type((weight, x)), act, weight, x, bias, cdims)
    __reset_BLAS_threads(old_threads)
    return ret
end

@stable default_mode="disable" function __fused_conv_bias_activation_impl(
        ::Type{T}, act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Optional{<:AbstractVector}, cdims::ConvDims) where {T, wT, xT, N, F}
    return __conv_bias_act(x, weight, cdims, bias, act)
end

function CRC.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(__fused_conv_bias_activation_impl),
        ::Type{DT}, act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Optional{<:AbstractVector}, cdims::ConvDims) where {DT, wT, xT, N, F}
    T = __get_concrete_fba_output_eltype(act, weight, x, bias)
    proj_w = CRC.ProjectTo(weight)
    proj_x = CRC.ProjectTo(x)
    proj_b = CRC.ProjectTo(bias)

    if __no_intermediate_needed(act, T)
        y = __conv_bias_act(x, weight, cdims, bias, act)
        ∇__fused_conv_bias_activation_impl_no_cached = @closure Δ -> begin
            old_threads = __maybe_reduce_BLAS_threads(weight)
            Δ = CRC.unthunk(NNlib.colmajor(Δ))
            ∂y = act === identity ? Δ : __activation_gradient(Δ, y, act, NotaNumber())
            ∂w, ∂x, ∂b = __conv_bias_partials(∂y, weight, x, bias, cdims)
            __reset_BLAS_threads(old_threads)
            return (∂∅, ∂∅, ∂∅, proj_w(∂w), proj_x(∂x), proj_b(∂b), ∂∅)
        end
        return y, ∇__fused_conv_bias_activation_impl_no_cached
    end

    # In any case here we need the intermediate pre-activation values
    y = similar(x, T, NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, N))
    __conv!(y, x, weight, cdims)

    if __needs_intermediate_but_has_rrule(act, T)
        z, y = __apply_bias_activation_cached!!(act, y, bias)
        ∇__fused_conv_bias_activation_impl_cached_crc = @closure Δ -> begin
            old_threads = __maybe_reduce_BLAS_threads(weight)
            Δ = CRC.unthunk(NNlib.colmajor(Δ))
            ∂y = __activation_gradient(Δ, z, act, y)
            ∂w, ∂x, ∂b = __conv_bias_partials(∂y, weight, x, bias, cdims)
            __reset_BLAS_threads(old_threads)
            return (∂∅, ∂∅, ∂∅, proj_w(∂w), proj_x(∂x), proj_b(∂b), ∂∅)
        end
        return z, ∇__fused_conv_bias_activation_impl_cached_crc
    end

    z, pb_f = CRC.rrule_via_ad(cfg, __bias_activation_impl, act, y, bias)
    ∇__fused_conv_bias_activation_impl_cached = @closure Δ -> begin
        old_threads = __maybe_reduce_BLAS_threads(weight)
        Δ = NNlib.colmajor(Δ)
        _, _, ∂y, ∂b = pb_f(Δ)
        ∂w, ∂x, _ = __conv_bias_partials(∂y, ∂b, weight, x, bias, cdims)
        __reset_BLAS_threads(old_threads)
        return (∂∅, ∂∅, ∂∅, proj_w(∂w), proj_x(∂x), proj_b(∂b), ∂∅)
    end

    return z, ∇__fused_conv_bias_activation_impl_cached
end

function __conv_bias_partials(∂y, weight, x, bias, cdims)
    return __conv_bias_partials(∂y, __added_bias_gradient(bias, ∂y), weight, x, bias, cdims)
end
function __conv_bias_partials(∂y, ∂b, weight, x, bias, cdims)
    ∂x = __∇conv_data(∂y, weight, cdims)
    ∂w = __∇conv_filter(x, ∂y, cdims)
    return ∂w, ∂x, ∂b
end

# Special handling for AMDGPU: AMDGPU doesn't support Float64 convolutions, so we need to
# type-cast everything
for (wT, xT) in [(Float64, Float64), (Float64, Float32), (Float32, Float64)],
    fname in (:__fused_conv_bias_activation_impl, :__generic_conv_bias_activation)

    for bT in (Float32, Float64)
        @eval begin
            function LuxLib.$fname(D::Type{<:AMDGPUDevice}, act::F,
                    weight::AbstractArray{$(wT), N}, x::AbstractArray{$(xT), N},
                    bias::Optional{<:AbstractVector{$(bT)}}, cdims::ConvDims) where {F, N}
                @warn "MIOpen doesn't support Float64 convolutions, type-casting \
                       everything to Float32 to avoid runtime errors" maxlog=1
                return _ofeltype_array(Float64,
                    LuxLib.$fname(D, act, _ofeltype_array(Float32, weight),
                        _ofeltype_array(Float32, x),
                        _ofeltype_array(Float32, bias), cdims))
            end

            CRC.@opt_out rrule(cfg::RuleConfig{>:HasReverseMode},
                ::typeof($fname), D::Type{<:AMDGPUDevice},
                act::F, weight::AbstractArray{$(wT), N}, x::AbstractArray{$(xT), N},
                bias::Optional{<:AbstractVector{$(bT)}}, cdims::ConvDims) where {F, N}
        end
    end

    @eval begin
        function LuxLib.$fname(
                D::Type{<:AMDGPUDevice}, act::F, weight::AbstractArray{$(wT), N},
                x::AbstractArray{$(xT), N}, ::Nothing, cdims::ConvDims) where {F, N}
            return _ofeltype_array(Float64,
                LuxLib.$fname(D, act, _ofeltype_array(Float32, weight),
                    _ofeltype_array(Float32, x), nothing, cdims))
        end

        CRC.@opt_out rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof($fname),
            D::Type{<:AMDGPUDevice}, act::F, weight::AbstractArray{$(wT), N},
            x::AbstractArray{$(xT), N}, ::Nothing, cdims::ConvDims) where {F, N}
    end
end

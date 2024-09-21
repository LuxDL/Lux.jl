function get_conv_input_weight(x, weight)
    return get_conv_input_weight(get_device_type((x, weight)), x, weight)
end

function get_conv_input_weight(::Type{Device}, x, weight) where {Device <: AbstractDevice}
    return get_conv_input_weight(
        Device, eltype_mismatch(safe_eltype(x), safe_eltype(weight)), x, weight)
end

function get_conv_input_weight(::Type{<:AbstractGPUDevice}, ::True, x, weight)
    T = promote_type(safe_eltype(x), safe_eltype(weight))
    safe_warning("Mixed Precision Inputs received for GPU convolution [weight: \
                 $(safe_eltype(weight))] and [x: $(safe_eltype(x))]. Promoting to $(T).", 1)
    return contiguous(ofeltype_array(T, x)), contiguous(ofeltype_array(T, weight))
end

function get_conv_input_weight(::Type{<:AbstractGPUDevice}, ::False, x, weight)
    return contiguous(x), contiguous(weight)
end

get_conv_input_weight(::Type{<:AbstractDevice}, ::StaticBool, x, weight) = x, weight

# Define some wrappers over NNlib operations. Useful once we ship our own versions
# with Kernel Abstractions and Loop Vectorization
function conv!(y, x, weight, cdims::ConvDims)
    return conv!(y, get_device_type((y, x, weight)), x, weight, cdims)
end
function conv!(y::AbstractArray{yT, N}, ::Type{<:AbstractDevice},
        x::AbstractArray{xT, N}, weight::AbstractArray{wT, N},
        cdims::ConvDims) where {yT, xT, wT, N}
    NNlib.conv!(y, x, weight, cdims)
    return
end
function conv!(y::AbstractArray{yT, N}, ::Type{<:Union{CUDADevice, AMDGPUDevice}},
        x::AbstractArray{xT, N}, weight::AbstractArray{wT, N},
        cdims::ConvDims) where {yT, xT, wT, N}
    if xT !== wT !== yT
        safe_warning(
            "Mixed Precision Inputs received for GPU convolution [weight: $(wT)] and \
             [x: $(xT)]. Promoting to $(yT).", 1)
    end
    NNlib.conv!(y, contiguous(ofeltype_array(yT, x)),
        contiguous(ofeltype_array(yT, weight)), cdims)
    return
end
function conv!(y::AbstractArray{yT, N}, dev::Type{<:AbstractGPUDevice},
        x::AbstractArray{xT, N}, weight::AbstractArray{wT, N},
        cdims::ConvDims) where {yT, xT, wT, N}
    if xT !== wT !== yT
        safe_warning(
            "Mixed Precision Inputs received for GPU convolution [weight: $(wT)] and \
             [x: $(xT)]. Promoting to $(yT).", 1)
    end
    x_cont = contiguous(ofeltype_array(yT, x))
    weight_cont = contiguous(ofeltype_array(yT, weight))
    fallback_slow_conv!(y, dev, x_cont, weight_cont, cdims)
    return
end

function fallback_slow_conv!(y::AbstractArray{yT, N}, dev::Type{<:AbstractDevice},
        x::AbstractArray{xT, N}, weight::AbstractArray{wT, N},
        cdims::ConvDims) where {yT, xT, wT, N}
    @warn "Falling back to slow convolution routine for $(dev) with x: size = \
           $(size(x)) eltype = $(xT) and weight: size = $(size(weight)) \
           eltype = $(wT)." maxlog=1
    # TODO: We should be able to reuse `y` for some part here for some efficiency
    @assert NNlib.groupcount(cdims)==1 "Only groups=1 is supported for now." # FIXME
    tmp = NNlib.unfold(x, cdims)
    weight_compact = reshape(weight, :, size(weight, N), 1)
    res = batched_matmul(tmp, weight_compact)
    copyto!(y, reshape(res, size(y)))
    return
end

conv(x, weight, cdims::ConvDims) = conv(get_device_type((x, weight)), x, weight, cdims)

function conv(::Type{Union{<:CPUDevice, <:CUDADevice, <:AMDGPUDevice}},
        x′, weight′, cdims::ConvDims)
    x, weight = get_conv_input_weight(x′, weight′)
    return NNlib.conv(x, weight, cdims)
end
function conv(dev::Type{<:AbstractDevice}, x′, weight′, cdims::ConvDims)
    x, weight = get_conv_input_weight(dev, x′, weight′)
    return fallback_slow_conv(dev, x, weight, cdims)
end

function fallback_slow_conv(dev, x, weight, cdims::ConvDims)
    y = similar(x, promote_type(eltype(x), eltype(weight)), NNlib.output_size(cdims)...,
        NNlib.channels_out(cdims), size(x, ndims(x)))
    fallback_slow_conv!(y, dev, x, weight, cdims)
    return y
end

function ∇conv_data(x′, weight′, cdims::ConvDims)
    x, weight = get_conv_input_weight(x′, weight′)
    return NNlib.∇conv_data(x, weight, cdims)
end

function ∇conv_filter(x′, y′, cdims::ConvDims)
    x, y = get_conv_input_weight(x′, y′)
    return NNlib.∇conv_filter(x, y, cdims)
end

function conv_bias_act(x′, weight′, cdims::ConvDims, bias′, act::F) where {F}
    x, weight = get_conv_input_weight(x′, weight′)
    bias = ofeltype_array(promote_type(safe_eltype(x), safe_eltype(weight)), bias′)
    return conv_bias_act(get_device_type((x, weight, bias)), x, weight, cdims, bias, act)
end

function conv_bias_act(::Type, x, weight, cdims, bias, act::F) where {F}
    y = similar(x, concrete_bias_act_output_eltype(act, weight, x, bias),
        NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, ndims(x)))
    conv!(y, x, weight, cdims)
    bias_activation!(y, internal_operation_mode((y, bias)), act, y, bias)
    return y
end

function conv_bias_act(::Type{CUDADevice}, x, weight, cdims, ::Nothing, act::F) where {F}
    return activation!!(act, conv(x, weight, cdims))
end
function conv_bias_act(::Type{CUDADevice}, x, weight, cdims, bias′, act::F) where {F}
    if act === identity || act === NNlib.relu
        bias = reshape_bias(x, bias′)
        return NNlib.conv_bias_act(x, weight, cdims, bias, act)
    end
    return conv_bias_act(Nothing, x, weight, cdims, bias′, act)
end

# Entry Points
function fused_conv(
        act::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Optional{<:AbstractVector}, cdims::ConvDims) where {F, wT, xT, N}
    old_threads = maybe_reduce_BLAS_threads(weight)
    y = fused_conv(internal_operation_mode((weight, x, bias)), act, weight, x, bias, cdims)
    reset_BLAS_threads(old_threads)
    return y
end

function fused_conv(::GenericBroadcastOp, act::F, weight::AbstractArray{wT, N},
        x::AbstractArray{xT, N}, bias::Optional{<:AbstractVector},
        cdims::ConvDims) where {F, wT, xT, N}
    return bias_activation(act, conv(x, weight, cdims), bias)
end

@stable default_mode="disable" function fused_conv(::AbstractInternalArrayOpMode, act::F,
        weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Optional{<:AbstractVector}, cdims::ConvDims) where {F, wT, xT, N}
    return conv_bias_act(x, weight, cdims, bias, act)
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(fused_conv),
        opmode::AbstractInternalArrayOpMode, act::F,
        weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
        bias::Optional{<:AbstractVector}, cdims::ConvDims) where {F, wT, xT, N}
    T = concrete_bias_act_output_eltype(act, weight, x, bias)
    𝒫w, 𝒫x, 𝒫b = CRC.ProjectTo(weight), CRC.ProjectTo(x), CRC.ProjectTo(bias)

    if unsafe_known(activation_intermediate_not_needed(act, T))
        y = conv_bias_act(x, weight, cdims, bias, act)
        ∇fused_conv_no_cached = @closure Δ -> begin
            return ∇fused_conv(
                Δ, weight, x, bias, cdims, y, NotaNumber(), 𝒫w, 𝒫x, 𝒫b, act)
        end
        return y, ∇fused_conv_no_cached
    end

    # In any case here we need the intermediate pre-activation values
    y = similar(x, T, NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, N))
    conv!(y, x, weight, cdims)

    if unsafe_known(activation_has_rrule(act, T))
        z, tmp = bias_activation_cached!!(act, y, bias)
        ∇fused_conv_cached = @closure Δ -> begin
            return ∇fused_conv(Δ, weight, x, bias, cdims, z, tmp, 𝒫w, 𝒫x, 𝒫b, act)
        end
        return z, ∇fused_conv_cached
    end

    z, ∇bias_activation = CRC.rrule_via_ad(cfg, bias_activation, act, y, bias)
    ∇fused_conv_cached = @closure Δ -> begin
        old_threads = maybe_reduce_BLAS_threads(weight)
        Δ = NNlib.colmajor(Δ)
        _, _, ∂y, ∂b = ∇bias_activation(Δ)
        ∂w, ∂x, _ = ∇conv_bias(∂y, ∂b, weight, x, bias, cdims)
        reset_BLAS_threads(old_threads)
        return ∂∅, ∂∅, ∂∅, 𝒫w(∂w), 𝒫x(∂x), 𝒫b(∂b), ∂∅
    end

    return z, ∇fused_conv_cached
end

CRC.@opt_out rrule(
    ::RuleConfig{>:HasReverseMode}, ::typeof(fused_conv), ::GenericBroadcastOp,
    ::F, ::AbstractArray{wT, N}, ::AbstractArray{xT, N},
    ::Optional{<:AbstractVector}, ::ConvDims) where {F, wT, xT, N}

function ∇fused_conv(Δ′, weight, x, bias, cdims::ConvDims, z, tmp, 𝒫w, 𝒫x, 𝒫b, act)
    old_threads = maybe_reduce_BLAS_threads(weight)
    Δ = CRC.unthunk(NNlib.colmajor(Δ′))
    ∂y = ∇activation(Δ, z, act, tmp)
    ∂w, ∂x, ∂b = ∇conv_bias(∂y, weight, x, bias, cdims)
    reset_BLAS_threads(old_threads)
    return ∂∅, ∂∅, ∂∅, 𝒫w(∂w), 𝒫x(∂x), 𝒫b(∂b), ∂∅
end

function ∇conv_bias(∂y, weight, x, bias, cdims::ConvDims)
    return ∇conv_bias(∂y, ∇bias_add(bias, ∂y), weight, x, bias, cdims)
end
function ∇conv_bias(∂y, ∂b, weight, x, _, cdims::ConvDims)
    return ∇conv_filter(x, ∂y, cdims), ∇conv_data(∂y, weight, cdims), ∂b
end

# Special handling for AMDGPU: AMDGPU doesn't support Float64 convolutions, so we need to
# type-cast everything
for (wT, xT) in [(Float64, Float64), (Float64, Float32), (Float32, Float64)]
    for bT in (Float32, Float64)
        @eval begin
            function fused_conv(opmode::GPUBroadcastOp{AMDGPUDevice}, act::F,
                    weight::AbstractArray{$(wT), N}, x::AbstractArray{$(xT), N},
                    bias::AbstractVector{$(bT)}, cdims::ConvDims) where {F, N}
                @warn "MIOpen doesn't support Float64 convolutions, type-casting \
                       everything to Float32 to avoid runtime errors" maxlog=1
                return ofeltype_array(Float64,
                    fused_conv(opmode, act, ofeltype_array(Float32, weight),
                        ofeltype_array(Float32, x), ofeltype_array(Float32, bias), cdims))
            end

            CRC.@opt_out rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(fused_conv),
                opmode::GPUBroadcastOp{AMDGPUDevice}, act::F,
                weight::AbstractArray{$(wT), N}, x::AbstractArray{$(xT), N},
                bias::Optional{<:AbstractVector{$(bT)}}, cdims::ConvDims) where {F, N}
        end
    end

    @eval begin
        function fused_conv(opmode::GPUBroadcastOp{AMDGPUDevice}, act::F,
                weight::AbstractArray{$(wT), N}, x::AbstractArray{$(xT), N},
                ::Nothing, cdims::ConvDims) where {F, N}
            return ofeltype_array(Float64,
                fused_conv(opmode, act, ofeltype_array(Float32, weight),
                    ofeltype_array(Float32, x), nothing, cdims))
        end

        CRC.@opt_out rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(fused_conv),
            opmode::GPUBroadcastOp{AMDGPUDevice}, act::F, weight::AbstractArray{$(wT), N},
            x::AbstractArray{$(xT), N}, ::Nothing, cdims::ConvDims) where {F, N}
    end
end

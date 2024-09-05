module LuxFluxExt

using ArgCheck: @argcheck
using Flux: Flux

using Lux: Lux, FluxModelConversionException, LuxOps

function Lux.convert_flux_model(l::T; preserve_ps_st::Bool=false, kwargs...) where {T}
    @warn "Transformation for type $T not implemented. Using `FluxLayer` as a fallback." maxlog=1

    if !preserve_ps_st
        @warn "`FluxLayer` uses the parameters and states of the `layer`. It is not \
                possible to NOT preserve the parameters and states. Ignoring this keyword \
                argument." maxlog=1
    end

    return Lux.FluxLayer(l)
end

Lux.convert_flux_model(l::Function; kwargs...) = Lux.WrappedFunction(l)

function Lux.convert_flux_model(l::Flux.Chain; kwargs...)
    fn = x -> Lux.convert_flux_model(x; kwargs...)
    layers = map(fn, l.layers)
    layers isa NamedTuple && return Lux.Chain(layers)
    return Lux.Chain(layers...)
end

function Lux.convert_flux_model(l::Flux.Dense; preserve_ps_st::Bool=false, kwargs...)
    out_dims, in_dims = size(l.weight)
    if preserve_ps_st
        bias = l.bias isa Bool ? nothing : copy(l.bias)
        return Lux.Dense(in_dims => out_dims, l.σ; init_weight=Returns(copy(l.weight)),
            init_bias=Returns(bias), use_bias=!(l.bias isa Bool))
    else
        return Lux.Dense(in_dims => out_dims, l.σ; use_bias=!(l.bias isa Bool))
    end
end

function Lux.convert_flux_model(l::Flux.Scale; preserve_ps_st::Bool=false, kwargs...)
    if preserve_ps_st
        return Lux.Scale(size(l.scale), l.σ; init_weight=Returns(copy(l.scale)),
            init_bias=Returns(copy(l.bias)), use_bias=!(l.bias isa Bool))
    else
        return Lux.Scale(size(l.scale), l.σ; use_bias=!(l.bias isa Bool))
    end
end

function Lux.convert_flux_model(l::Flux.Maxout; kwargs...)
    return Lux.Maxout(Lux.convert_flux_model.(l.layers; kwargs...)...)
end

function Lux.convert_flux_model(l::Flux.SkipConnection; kwargs...)
    connection = l.connection isa Function ? l.connection :
                 Lux.convert_flux_model(l.connection; kwargs...)
    return Lux.SkipConnection(Lux.convert_flux_model(l.layers; kwargs...), connection)
end

function Lux.convert_flux_model(l::Flux.Bilinear; preserve_ps_st::Bool=false, kwargs...)
    out, in1, in2 = size(l.weight)
    if preserve_ps_st
        return Lux.Bilinear((in1, in2) => out, l.σ; init_weight=Returns(copy(l.weight)),
            init_bias=Returns(copy(l.bias)), use_bias=!(l.bias isa Bool))
    else
        return Lux.Bilinear((in1, in2) => out, l.σ; use_bias=!(l.bias isa Bool))
    end
end

function Lux.convert_flux_model(l::Flux.Parallel; kwargs...)
    fn = x -> Lux.convert_flux_model(x; kwargs...)
    layers = map(fn, l.layers)
    if layers isa NamedTuple
        return Lux.Parallel(l.connection; layers...)
    else
        return Lux.Parallel(l.connection, layers...)
    end
end

function Lux.convert_flux_model(l::Flux.PairwiseFusion; kwargs...)
    @warn "Flux.PairwiseFusion and Lux.PairwiseFusion are semantically different. Using \
           `FluxLayer` as a fallback." maxlog=1
    return Lux.FluxLayer(l)
end

function Lux.convert_flux_model(l::Flux.Embedding; preserve_ps_st::Bool=true, kwargs...)
    out_dims, in_dims = size(l.weight)
    if preserve_ps_st
        return Lux.Embedding(in_dims => out_dims; init_weight=Returns(copy(l.weight)))
    else
        return Lux.Embedding(in_dims => out_dims)
    end
end

function Lux.convert_flux_model(l::Flux.Conv; preserve_ps_st::Bool=false, kwargs...)
    k = size(l.weight)[1:(end - 2)]
    in_chs, out_chs = size(l.weight)[(end - 1):end]
    groups = l.groups
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    if preserve_ps_st
        _bias = l.bias isa Bool ? nothing : vec(copy(l.bias))
        return Lux.Conv(k, in_chs * groups => out_chs, l.σ; l.stride, pad, l.dilation,
            groups, init_weight=Returns(Lux.maybe_flip_conv_weight(l.weight)),
            init_bias=Returns(_bias), use_bias=!(l.bias isa Bool))
    else
        return Lux.Conv(k, in_chs * groups => out_chs, l.σ; l.stride, pad,
            l.dilation, groups, use_bias=!(l.bias isa Bool))
    end
end

function Lux.convert_flux_model(
        l::Flux.ConvTranspose; preserve_ps_st::Bool=false, kwargs...)
    k = size(l.weight)[1:(end - 2)]
    out_chs, in_chs = size(l.weight)[(end - 1):end]
    groups = l.groups
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    if preserve_ps_st
        _bias = l.bias isa Bool ? nothing : vec(copy(l.bias))
        return Lux.ConvTranspose(k, in_chs * groups => out_chs, l.σ; l.stride,
            pad, l.dilation, groups, use_bias=!(l.bias isa Bool),
            init_weight=Returns(Lux.maybe_flip_conv_weight(l.weight)),
            init_bias=Returns(_bias))
    else
        return Lux.ConvTranspose(k, in_chs * groups => out_chs, l.σ; l.stride, pad,
            l.dilation, groups, use_bias=!(l.bias isa Bool))
    end
end

function Lux.convert_flux_model(l::Flux.CrossCor; preserve_ps_st::Bool=false, kwargs...)
    k = size(l.weight)[1:(end - 2)]
    in_chs, out_chs = size(l.weight)[(end - 1):end]
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    if preserve_ps_st
        _bias = l.bias isa Bool ? nothing : vec(copy(l.bias))
        return Lux.Conv(k, in_chs => out_chs, l.σ; l.stride, pad, l.dilation,
            init_weight=Returns(copy(l.weight)), init_bias=Returns(_bias),
            use_bias=!(l.bias isa Bool), cross_correlation=true)
    else
        return Lux.Conv(k, in_chs => out_chs, l.σ; l.stride, pad, l.dilation,
            use_bias=!(l.bias isa Bool), cross_correlation=true)
    end
end

Lux.convert_flux_model(l::Flux.AdaptiveMaxPool; kwargs...) = Lux.AdaptiveMaxPool(l.out)

Lux.convert_flux_model(l::Flux.AdaptiveMeanPool; kwargs...) = Lux.AdaptiveMeanPool(l.out)

Lux.convert_flux_model(::Flux.GlobalMaxPool; kwargs...) = Lux.GlobalMaxPool()

Lux.convert_flux_model(::Flux.GlobalMeanPool; kwargs...) = Lux.GlobalMeanPool()

function Lux.convert_flux_model(l::Flux.MaxPool; kwargs...)
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    return Lux.MaxPool(l.k; l.stride, pad)
end

function Lux.convert_flux_model(l::Flux.MeanPool; kwargs...)
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    return Lux.MeanPool(l.k; l.stride, pad)
end

Lux.convert_flux_model(l::Flux.Dropout; kwargs...) = Lux.Dropout(l.p; l.dims)

function Lux.convert_flux_model(l::Flux.LayerNorm; kwargs...)
    @warn "Flux.LayerNorm and Lux.LayerNorm are semantically different specifications. \
           Using `FluxLayer` as a fallback." maxlog=1
    return Lux.FluxLayer(l)
end

Lux.convert_flux_model(::typeof(identity); kwargs...) = Lux.NoOpLayer()

Lux.convert_flux_model(::typeof(Flux.flatten); kwargs...) = Lux.FlattenLayer()

Lux.convert_flux_model(l::Flux.PixelShuffle; kwargs...) = Lux.PixelShuffle(l.r)

function Lux.convert_flux_model(l::Flux.Upsample{mode}; kwargs...) where {mode}
    return Lux.Upsample(mode; l.scale, l.size, align_corners=false)
end

function Lux.convert_flux_model(
        l::Flux.RNNCell; preserve_ps_st::Bool=false, force_preserve::Bool=false)
    out_dims, in_dims = size(l.Wi)
    if preserve_ps_st
        if force_preserve
            throw(FluxModelConversionException("Recurrent Cell: $(typeof(l)) for Flux uses a `reset!` mechanism which hasn't been extensively tested with `FluxLayer`. Rewrite the model manually to use `RNNCell`."))
        end
        @warn "Preserving Parameters: `Wh` & `Wi` for `Flux.RNNCell` is ambiguous in Lux and hence not supported. Ignoring these parameters." maxlog=1
        return Lux.RNNCell(in_dims => out_dims, l.σ; init_bias=Returns(copy(l.b)),
            init_state=Returns(copy(l.state0)))
    else
        return Lux.RNNCell(in_dims => out_dims, l.σ)
    end
end

function Lux.convert_flux_model(
        l::Flux.LSTMCell; preserve_ps_st::Bool=false, force_preserve::Bool=false)
    _out_dims, in_dims = size(l.Wi)
    out_dims = _out_dims ÷ 4
    if preserve_ps_st
        if force_preserve
            throw(FluxModelConversionException("Recurrent Cell: $(typeof(l)) for Flux uses a `reset!` mechanism which hasn't been extensively tested with `FluxLayer`. Rewrite the model manually to use `LSTMCell`."))
        end
        @warn "Preserving Parameters: `Wh` & `Wi` for `Flux.LSTMCell` is ambiguous in Lux \
               and hence not supported. Ignoring these parameters." maxlog=1
        bs = LuxOps.multigate(l.b, Val(4))
        _s, _m = copy.(l.state0)
        return Lux.LSTMCell(in_dims => out_dims; init_bias=Returns.(bs),
            init_state=Returns(_s), init_memory=Returns(_m))
    else
        return Lux.LSTMCell(in_dims => out_dims)
    end
end

function Lux.convert_flux_model(
        l::Flux.GRUCell; preserve_ps_st::Bool=false, force_preserve::Bool=false)
    _out_dims, in_dims = size(l.Wi)
    out_dims = _out_dims ÷ 3
    if preserve_ps_st
        if force_preserve
            throw(FluxModelConversionException("Recurrent Cell: $(typeof(l)) for Flux uses a `reset!` mechanism which hasn't been extensively tested with `FluxLayer`. Rewrite the model manually to use `GRUCell`."))
        end
        @warn "Preserving Parameters: `Wh` & `Wi` for `Flux.GRUCell` is ambiguous in Lux \
               and hence not supported. Ignoring these parameters." maxlog=1
        bs = LuxOps.multigate(l.b, Val(3))
        return Lux.GRUCell(
            in_dims => out_dims; init_bias=Returns.(bs), init_state=Returns(copy(l.state0)))
    else
        return Lux.GRUCell(in_dims => out_dims)
    end
end

function Lux.convert_flux_model(
        l::Flux.BatchNorm; preserve_ps_st::Bool=false, force_preserve::Bool=false)
    if preserve_ps_st
        if l.track_stats
            force_preserve && return Lux.FluxLayer(l)
            @warn "Preserving the state of `Flux.BatchNorm` is currently not supported. \
                   Ignoring the state." maxlog=1
        end
        if l.affine
            return Lux.BatchNorm(
                l.chs, l.λ; l.affine, l.track_stats, epsilon=l.ϵ, l.momentum,
                init_bias=Returns(copy(l.β)), init_scale=Returns(copy(l.γ)))
        else
            return Lux.BatchNorm(
                l.chs, l.λ; l.affine, l.track_stats, epsilon=l.ϵ, l.momentum)
        end
    end
    return Lux.BatchNorm(l.chs, l.λ; l.affine, l.track_stats, epsilon=l.ϵ, l.momentum)
end

function Lux.convert_flux_model(
        l::Flux.GroupNorm; preserve_ps_st::Bool=false, force_preserve::Bool=false)
    if preserve_ps_st
        @argcheck !l.track_stats
        if l.affine
            return Lux.GroupNorm(l.chs, l.G, l.λ; l.affine, epsilon=l.ϵ,
                init_bias=Returns(copy(l.β)), init_scale=Returns(copy(l.γ)))
        else
            return Lux.GroupNorm(l.chs, l.G, l.λ; l.affine, epsilon=l.ϵ)
        end
    end
    return Lux.GroupNorm(l.chs, l.G, l.λ; l.affine, epsilon=l.ϵ)
end

const _INVALID_TRANSFORMATION_TYPES = Union{<:Flux.Recur}

function Lux.convert_flux_model(l::T; kwargs...) where {T <: _INVALID_TRANSFORMATION_TYPES}
    throw(FluxModelConversionException("Transformation of type $(T) is not supported."))
end

end

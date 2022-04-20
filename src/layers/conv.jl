struct Conv{N,bias,cdims,M,F1,F2} <: AbstractExplicitLayer
    λ::F1
    in_chs::Int
    out_chs::Int
    kernel_size::NTuple{N,Int}
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
    initW::F2
end

function Conv(
    k::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer},
    λ=identity;
    input_size::Union{Nothing,NTuple{N,Integer}}=nothing,
    initW=glorot_uniform,
    stride=1,
    pad=0,
    dilation=1,
    groups=1,
    bias=true,
) where {N}
    stride = expand(Val(N), stride)
    dilation = expand(Val(N), dilation)
    pad = calc_padding(Conv, pad, k, dilation, stride)
    λ = NNlib.fast_act(λ)
    cdims = if input_size === nothing
        nothing
    else
        DenseConvDims(
            (input_size..., first(ch), 1),
            (k..., ch...);
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
        )
    end
    return Conv{N,bias,cdims,length(pad),typeof(λ),typeof(initW)}(
        λ, first(ch), last(ch), k, stride, pad, dilation, groups, initW
    )
end

function initialparameters(rng::AbstractRNG, c::Conv{N,bias}) where {N,bias}
    initW(args...) = c.initW(rng, args...)
    weight = convfilter(c.kernel_size, c.in_chs => c.out_chs; init=initW, groups=c.groups)
    return ComponentArray(
        bias ? (weight=weight, bias=zeros(eltype(weight), ntuple(_ -> 1, N)..., c.out_chs, 1)) : (weight=weight,)
    )
end

parameterlength(c::Conv{N,bias}) where {N,bias} = prod(c.kernel_size) * c.in_chs * c.out_chs + (bias ? c.out_chs : 0)

Base.@pure function (c::Conv{N,bias,C})(
    x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {N,bias,C}
    cdims = if C === nothing
        DenseConvDims(x, ps.weight; stride=c.stride, padding=c.pad, dilation=c.dilation, groups=c.groups)
    else
        C
    end
    if bias
        return c.λ.(conv_wrapper(x, ps.weight, cdims) .+ ps.bias), st
        # FIXME: Needs https://github.com/FluxML/NNlibCUDA.jl/pull/45 to be merged
        # return conv_bias_act(x, ps.weight, cdims, ps.bias, λ), st
    else
        return c.λ.(conv_wrapper(x, ps.weight, cdims)), st
    end
end

function Base.show(io::IO, l::Conv)
    print(io, "Conv(", l.kernel_size)
    print(io, ", ", l.in_chs, " => ", l.out_chs)
    _print_conv_opt(io, l)
    return print(io, ")")
end

function _print_conv_opt(io::IO, l::Conv{bias}) where {bias}
    l.λ == identity || print(io, ", ", l.λ)
    all(==(0), l.pad) || print(io, ", pad=", _maybetuple_string(l.pad))
    all(==(1), l.stride) || print(io, ", stride=", _maybetuple_string(l.stride))
    all(==(1), l.dilation) || print(io, ", dilation=", _maybetuple_string(l.dilation))
    (l.groups == 1) || print(io, ", groups=", l.groups)
    (bias == false) && print(io, ", bias=false")
    return nothing
end

struct MaxPool{N,M,pdims} <: AbstractExplicitLayer
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
end

function MaxPool(k::NTuple{N,Integer}; pad=0, stride=k, input_size::Union{Nothing,NTuple{N,Integer}}=nothing) where {N}
    stride = expand(Val(N), stride)
    pad = calc_padding(MaxPool, pad, k, 1, stride)
    pdims = if input_size === nothing
        nothing
    else
        PoolDims((input_size..., first(ch), 1), k; stride=stride, padding=pad, dilation=dilation)
    end
    return MaxPool{N,length(pad),pdims}(k, pad, stride)
end

Base.@pure function (m::MaxPool{N,M,P})(x, ps, st::NamedTuple) where {N,M,P}
    pdims = P === nothing ? PoolDims(x, m.k; padding=m.pad, stride=m.stride) : P
    return maxpool(x, pdims), st
end

function Base.show(io::IO, m::MaxPool)
    print(io, "MaxPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    return print(io, ")")
end

struct MeanPool{N,M,pdims} <: AbstractExplicitLayer
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
end

function MeanPool(k::NTuple{N,Integer}; pad=0, stride=k, input_size::Union{Nothing,NTuple{N,Integer}}=nothing) where {N}
    stride = expand(Val(N), stride)
    pad = calc_padding(MeanPool, pad, k, 1, stride)
    pdims = if input_size === nothing
        nothing
    else
        PoolDims((input_size..., first(ch), 1), k; stride=stride, padding=pad, dilation=dilation)
    end
    return MeanPool{N,length(pad),pdims}(k, pad, stride)
end

Base.@pure function (m::MeanPool{N,M,P})(x, ps, st::NamedTuple) where {N,M,P}
    pdims = P === nothing ? PoolDims(x, m.k; padding=m.pad, stride=m.stride) : P
    return meanpool(x, pdims), st
end

function Base.show(io::IO, m::MeanPool)
    print(io, "MeanPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    return print(io, ")")
end

# Upsampling
struct Upsample{mode,S,T} <: AbstractExplicitLayer
    scale::S
    size::T
end

function Upsample(mode::Symbol=:nearest; scale=nothing, size=nothing)
    mode in [:nearest, :bilinear, :trilinear] || throw(ArgumentError("mode=:$mode is not supported."))
    if !(isnothing(scale) ⊻ isnothing(size))
        throw(ArgumentError("Either scale or size should be specified (but not both)."))
    end
    return Upsample{mode,typeof(scale),typeof(size)}(scale, size)
end

Upsample(scale, mode::Symbol=:nearest) = Upsample(mode; scale)

function (m::Upsample{:nearest})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_nearest(x, m.scale), st
end
function (m::Upsample{:nearest,Int})(
    x::AbstractArray{T,N}, ps, st::NamedTuple
) where {T,N}
    return NNlib.upsample_nearest(x, ntuple(i -> m.scale, N - 2)), st
end
function (m::Upsample{:nearest,Nothing})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_nearest(x; size=m.size), st
end

function (m::Upsample{:bilinear})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_bilinear(x, m.scale), st
end
function (m::Upsample{:bilinear,Nothing})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_bilinear(x; size=m.size), st
end

function (m::Upsample{:trilinear})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_trilinear(x, m.scale), st
end
function (m::Upsample{:trilinear,Nothing})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_trilinear(x; size=m.size), st
end

function Base.show(io::IO, u::Upsample{mode}) where {mode}
    print(io, "Upsample(")
    print(io, ":", mode)
    u.scale !== nothing && print(io, ", scale = $(u.scale)")
    u.size !== nothing && print(io, ", size = $(u.size)")
    return print(io, ")")
end

# Global Mean Pooling
struct GlobalMeanPool <: AbstractExplicitLayer end

Base.@pure function (g::GlobalMeanPool)(x, ps, st::NamedTuple)
    return meanpool(x, PoolDims(x, size(x)[1:(end - 2)])), st
end

# Global Max Pooling
struct GlobalMaxPool <: AbstractExplicitLayer end

Base.@pure function (g::GlobalMaxPool)(x, ps, st::NamedTuple)
    return maxpool(x, PoolDims(x, size(x)[1:(end - 2)])), st
end

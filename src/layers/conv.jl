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
    return (bias ? (weight=weight, bias=zeros(eltype(weight), ntuple(_ -> 1, N)..., c.out_chs, 1)) : (weight=weight,))
end

parameterlength(c::Conv{N,bias}) where {N,bias} = prod(c.kernel_size) * c.in_chs * c.out_chs + (bias ? c.out_chs : 0)

Base.@pure function (c::Conv{N,bias,C})(x::AbstractArray, ps::NamedTuple, st::NamedTuple) where {N,bias,C}
    cdims = if C === nothing
        DenseConvDims(x, ps.weight; stride=c.stride, padding=c.pad, dilation=c.dilation, groups=c.groups)
    else
        C
    end
    if bias
        return fast_conv_bias_act(x, ps.weight, cdims, ps.bias, c.λ), st
        # return c.λ.(conv(x, ps.weight, cdims) .+ ps.bias), st
        # FIXME: Needs https://github.com/FluxML/NNlibCUDA.jl/pull/45 to be merged
        # return conv_bias_act(x, ps.weight, cdims, ps.bias, λ), st
    else
        return c.λ.(conv(x, ps.weight, cdims)), st
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
    return (bias == false) && print(io, ", bias=false")
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

Base.@pure function (m::MaxPool{N,M,P})(x, ::NamedTuple, st::NamedTuple) where {N,M,P}
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

Base.@pure function (m::MeanPool{N,M,P})(x, ::NamedTuple, st::NamedTuple) where {N,M,P}
    pdims = P === nothing ? PoolDims(x, m.k; padding=m.pad, stride=m.stride) : P
    return meanpool(x, pdims), st
end

function Base.show(io::IO, m::MeanPool)
    print(io, "MeanPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    return print(io, ")")
end

struct Conv{N,M,F1,F2} <: ExplicitLayer
    λ::F1
    in_chs::Int
    out_chs::Int
    kernel_size::NTuple{N,Int}
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
    bias::Bool
    initW::F2
end

function Conv(
    k::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer},
    λ=identity;
    init=glorot_uniform,
    stride=1,
    pad=0,
    dilation=1,
    groups=1,
    bias=true,
) where {N}
    stride = expand(Val(N), stride)
    dilation = expand(Val(N), dilation)
    pad = calc_padding(Conv, pad, k, dilation, stride)
    return Conv(λ, first(ch), last(ch), k, stride, pad, dilation, groups, bias, init)
end

function initialparameters(rng::AbstractRNG, c::Conv{N}) where {N}
    initW(args...) = c.initW(rng, args...)
    weight = convfilter(c.kernel_size, c.in_chs => c.out_chs; init=initW, groups=c.groups)
    return (c.bias ? (weight=weight, bias=zeros(eltype(weight), ntuple(_ -> 1, N)..., c.out_chs, 1)) : (weight = weight,))
end

parameterlength(c::Conv) = prod(c.kernel_size) * c.in_chs * c.out_chs + (c.bias ? c.out_chs : 0)

function (c::Conv)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    λ = NNlib.fast_act(c.λ, x)
    cdims = DenseConvDims(x, ps.weight; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)
    if c.bias
        return conv_bias_act(x, ps.weight, cdims, ps.bias, λ), st
    else
        return λ.(conv(x, ps.weight, cdims)), st
    end
end

function Base.show(io::IO, l::Conv)
    print(io, "Conv(", l.kernel_size)
    print(io, ", ", l.in_chs, " => ", l.out_chs)
    _print_conv_opt(io, l)
    print(io, ")")
end
  
function _print_conv_opt(io::IO, l)
    l.λ == identity || print(io, ", ", l.λ)
    all(==(0), l.pad) || print(io, ", pad=", _maybetuple_string(l.pad))
    all(==(1), l.stride) || print(io, ", stride=", _maybetuple_string(l.stride))
    all(==(1), l.dilation) || print(io, ", dilation=", _maybetuple_string(l.dilation))
    (l.groups == 1) || print(io, ", groups=", l.groups)
    (l.bias == false) && print(io, ", bias=false")
end

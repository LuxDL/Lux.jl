"""
    Conv(filter, in => out, σ = identity; stride = 1, pad = 0, dilation = 1, groups = 1, [bias, initW])

Standard convolutional layer.

# Arguments

* `filter` is a tuple of integers specifying the size of the convolutional kernel
* `in` and `out` specify the number of input and output channels.

Image data should be stored in WHCN order (width, height, channels, batch). In other words, a 100×100 RGB image would be a `100×100×3×1` array, and a batch of 50 would be a `100×100×3×50` array. This has `N = 2` spatial dimensions, and needs a kernel size like `(5,5)`, a 2-tuple of integers. To take convolutions along `N` feature dimensions, this layer expects as input an array with `ndims(x) == N+2`, where `size(x, N+1) == in` is the number of input channels, and `size(x, ndims(x))` is (as always) the number of observations in a batch.
* `filter` should be a tuple of `N` integers.
* Keywords `stride` and `dilation` should each be either single integer, or a tuple with `N` integers.
* Keyword `pad` specifies the number of elements added to the borders of the data array. It can be
  - a single integer for equal padding all around,
  - a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
  - a tuple of `2*N` integers, for asymmetric padding, or
  - the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.
* Keyword `groups` is expected to be an `Int`. It specifies the number of groups to divide a convolution into.

Keywords to control initialization of the layer:
* `initW` - Function used to generate initial weights. Defaults to `glorot_uniform`.
* `bias` - The initial bias vector is all zero by default. Trainable bias can be disabled entirely by setting this to `false`.
"""
struct Conv{N,bias,M,F1,F2} <: AbstractExplicitLayer
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
    return Conv{N,bias,length(pad),typeof(λ),typeof(initW)}(
        λ, first(ch), last(ch), k, stride, pad, dilation, groups, initW
    )
end

function initialparameters(rng::AbstractRNG, c::Conv{N,bias}) where {N,bias}
    initW(args...) = c.initW(rng, args...)
    weight = convfilter(c.kernel_size, c.in_chs => c.out_chs; init=initW, groups=c.groups)
    return bias ? (weight=weight, bias=zeros(eltype(weight), ntuple(_ -> 1, N)..., c.out_chs, 1)) : (weight=weight,)
end

parameterlength(c::Conv{N,bias}) where {N,bias} = prod(c.kernel_size) * c.in_chs * c.out_chs ÷ c.groups  + (bias ? c.out_chs : 0)

function (c::Conv{N,bias})(
    x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {N,bias}
    cdims = DenseConvDims(x, ps.weight; stride=c.stride, padding=c.pad, dilation=c.dilation, groups=c.groups)
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

function _print_conv_opt(io::IO, l::Conv{N,bias}) where {N,bias}
    l.λ == identity || print(io, ", ", l.λ)
    all(==(0), l.pad) || print(io, ", pad=", _maybetuple_string(l.pad))
    all(==(1), l.stride) || print(io, ", stride=", _maybetuple_string(l.stride))
    all(==(1), l.dilation) || print(io, ", dilation=", _maybetuple_string(l.dilation))
    (l.groups == 1) || print(io, ", groups=", l.groups)
    (bias == false) && print(io, ", bias=false")
    return nothing
end

"""
    MaxPool(window::NTuple; pad=0, stride=window)

# Arguments

* Max pooling layer, which replaces all pixels in a block of size `window` with one.
* Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(window)`.
* By default the window size is also the stride in each dimension.
* The keyword `pad` accepts the same options as for the [`Conv`](@ref) layer, including `SamePad()`.

See also [`Conv`](@ref), [`MeanPool`](@ref), [`GlobalMaxPool`](@ref).
"""
struct MaxPool{N,M} <: AbstractExplicitLayer
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
end

function MaxPool(k::NTuple{N,Integer}; pad=0, stride=k) where {N}
    stride = expand(Val(N), stride)
    pad = calc_padding(MaxPool, pad, k, 1, stride)
    return MaxPool{N,length(pad)}(k, pad, stride)
end

function (m::MaxPool{N,M})(x, ps, st::NamedTuple) where {N,M}
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return maxpool(x, pdims), st
end

function Base.show(io::IO, m::MaxPool)
    print(io, "MaxPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    return print(io, ")")
end


"""
    MeanPool(window::NTuple; pad=0, stride=window)

# Arguments

* Mean pooling layer, which replaces all pixels in a block of size `window` with one.
* Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(window)`.
* By default the window size is also the stride in each dimension.
* The keyword `pad` accepts the same options as for the [`Conv`](@ref) layer, including `SamePad()`.

See also [`Conv`](@ref), [`MaxPool`](@ref), [`GlobalMeanPool`](@ref).
"""
struct MeanPool{N,M} <: AbstractExplicitLayer
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
end

function MeanPool(k::NTuple{N,Integer}; pad=0, stride=k) where {N}
    stride = expand(Val(N), stride)
    pad = calc_padding(MeanPool, pad, k, 1, stride)
    return MeanPool{N,length(pad)}(k, pad, stride)
end

function (m::MeanPool{N,M})(x, ps, st::NamedTuple) where {N,M}
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return meanpool(x, pdims), st
end

function Base.show(io::IO, m::MeanPool)
    print(io, "MeanPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    return print(io, ")")
end

"""
    Upsample(mode = :nearest; [scale, size]) 
    Upsample(scale, mode = :nearest)  

An upsampling layer.

# Arguments

One of two keywords must be given:
* If `scale` is a number, this applies to all but the last two dimensions (channel and batch) of the input.  It may also be a tuple, to control dimensions individually.
 * Alternatively, keyword `size` accepts a tuple, to directly specify the leading dimensions of the output.

Currently supported upsampling `mode`s and corresponding NNlib's methods are:
  - `:nearest` -> [`NNlib.upsample_nearest`](@ref) 
  - `:bilinear` -> [`NNlib.upsample_bilinear`](@ref)
  - `:trilinear` -> [`NNlib.upsample_trilinear`](@ref)
"""
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

"""
    GlobalMeanPool()

Global Mean Pooling layer. Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output, by performing max pooling on the complete (w,h)-shaped feature maps.

See also [`MeanPool`](@ref), [`GlobalMaxPool`](@ref).
"""
struct GlobalMeanPool <: AbstractExplicitLayer end

function (g::GlobalMeanPool)(x, ps, st::NamedTuple)
    return meanpool(x, PoolDims(x, size(x)[1:(end - 2)])), st
end

"""
    GlobalMaxPool()

Global Max Pooling layer. Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output, by performing max pooling on the complete (w,h)-shaped feature maps.

See also [`MaxPool`](@ref), [`GlobalMeanPool`](@ref).
"""
struct GlobalMaxPool <: AbstractExplicitLayer end

function (g::GlobalMaxPool)(x, ps, st::NamedTuple)
    return maxpool(x, PoolDims(x, size(x)[1:(end - 2)])), st
end

"""
    AdaptiveMaxPool(out::NTuple)

Adaptive Max Pooling layer. Calculates the necessary window size such that its output has `size(y)[1:N] == out`. Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

See also [`MaxPool`](@ref), [`AdaptiveMeanPool`](@ref).
"""
struct AdaptiveMaxPool{S, O} <: AbstractExplicitLayer
    out::NTuple{O, Int}
    AdaptiveMaxPool(out::NTuple{O, Int}) where O = new{O + 2, O}(out)
end

function (a::AdaptiveMaxPool{S})(x::AbstractArray{T, S}, ps, st::NamedTuple) where {S, T}
    insize = size(x)[1:end-2]
    outsize = a.out
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return maxpool(x, pdims), st
end

function Base.show(io::IO, a::AdaptiveMaxPool)
    print(io, "AdaptiveMaxPool(", a.out, ")")
end

"""
    AdaptiveMeanPool(out::NTuple)

Adaptive Mean Pooling layer. Calculates the necessary window size such that its output has `size(y)[1:N] == out`. Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

See also [`MaxPool`](@ref), [`AdaptiveMaxPool`](@ref).
"""
struct AdaptiveMeanPool{S, O} <: AbstractExplicitLayer
    out::NTuple{O, Int}
    AdaptiveMeanPool(out::NTuple{O, Int}) where O = new{O + 2, O}(out)
end

function (a::AdaptiveMeanPool{S})(x::AbstractArray{T, S}, ps, st::NamedTuple) where {S, T}
    insize = size(x)[1:end-2]
    outsize = a.out
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return meanpool(x, pdims), st
end

function Base.show(io::IO, a::AdaptiveMeanPool)
    print(io, "AdaptiveMeanPool(", a.out, ")")
end
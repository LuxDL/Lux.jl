struct SamePad end

function calc_padding(pad, ::NTuple{N}, dilation, stride) where {N}
    return Utils.expand(Val(2 * N), pad)
end

function calc_padding(::SamePad, k::NTuple, dilation, stride)
    # Ref: "A guide to convolution arithmetic for deep learning"
    # https://arxiv.org/abs/1603.07285 Effective kernel size, including dilation
    k_eff = @. k + (k - 1) * (dilation - 1)
    # How much total padding needs to be applied?
    pad_amt = @. k_eff - 1
    # In case amount of padding is odd we need to apply different amounts to each side.
    return Tuple(mapfoldl(i -> [cld(i, 2), fld(i, 2)], vcat, pad_amt))
end

CRC.@non_differentiable calc_padding(::Any...)

function conv_transpose_dims(
        x::AbstractArray, weight::AbstractArray; padding, stride, dilation, groups)
    # Calculate size of "input", from ∇conv_data()'s perspective...
    function calc_dim(xsz, wsz, stride, dilation, pad)
        return (xsz - 1) * stride + 1 + (wsz - 1) * dilation - pad
    end
    combined_pad = ntuple(i -> padding[2i - 1] + padding[2i], length(padding) ÷ 2)
    I = map(calc_dim, size(x)[1:(end - 2)], size(weight)[1:(end - 2)],
        stride, dilation, combined_pad)
    C_in = size(weight)[end - 1] * groups
    C_out = size(weight)[end]
    batch_size = size(x)[end]
    w_size = size(weight)

    size(x)[end - 1] != C_out &&
        throw(DimensionMismatch(lazy"Expected $(C_out) input channels but got $(size(x)[end - 1]) channels."))

    # Create DenseConvDims() that looks like the corresponding conv()
    return DenseConvDims(
        (I..., C_in, batch_size), w_size; stride, padding, dilation, groups)
end

CRC.@non_differentiable conv_transpose_dims(::Any...)

conv_transpose(x, weight, cdims) = LuxLib.Impl.∇conv_data(x, weight, cdims)

function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
    insize = size(x)[1:(end - 2)]
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    return PoolDims(x, k; padding=0, stride=stride)
end

CRC.@non_differentiable compute_adaptive_pooling_dims(::Any, ::Any)

function init_conv_filter(rng::AbstractRNG, filter::NTuple{N, Integer},
        ch::Pair{<:Integer, <:Integer}; init=glorot_uniform, groups=1) where {N}
    cin, cout = ch
    @argcheck cin % groups==0 DimensionMismatch("Input channel dimension must be divisible by groups.")
    @argcheck cout % groups==0 DimensionMismatch("Output channel dimension must be divisible by groups.")
    return init(rng, filter..., cin ÷ groups, cout)
end

@doc doc"""
    Conv(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
         activation=identity; init_weight=glorot_uniform, init_bias=zeros32, stride=1,
         pad=0, dilation=1, groups=1, use_bias=True())

Standard convolutional layer.

!!! tip "Conv 2D"

    Image data should be stored in WHCN order (width, height, channels, batch). In other
    words, a `100 x 100` RGB image would be a `100 x 100 x 3 x 1` array, and a batch of 50
    would be a `100 x 100 x 3 x 50` array. This has `N = 2` spatial dimensions, and needs
    a kernel size like `(5, 5)`, a 2-tuple of integers. To take convolutions along `N`
    feature dimensions, this layer expects as input an array with `ndims(x) == N + 2`, where
    `size(x, N + 1) == in_chs` is the number of input channels, and `size(x, ndims(x))` is
    the number of observations in a batch.

!!! warning

    Frameworks like [`Pytorch`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
    perform cross-correlation in their convolution layers

## Arguments

  - `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D
         convolutions `length(k) == 2`
  - `in_chs`: Number of input channels
  - `out_chs`: Number of input and output channels
  - `activation`: Activation Function

# Extended Help

## Keyword Arguments

  - `init_weight`: Controls the initialization of the weight parameter
  - `init_bias`: Controls the initialization of the bias parameter

  - `stride`: Should each be either single integer, or a tuple with `N` integers
  - `dilation`: Should each be either single integer, or a tuple with `N` integers
  - `pad`: Specifies the number of elements added to the borders of the data array. It can
           be

      + a single integer for equal padding all around,
      + a tuple of `N` integers, to apply the same padding at begin/end of each spatial
        dimension,
      + a tuple of `2*N` integers, for asymmetric padding, or
      + the singleton `SamePad()`, to calculate padding such that
        `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial
        dimension.
      + Periodic padding can achieved by pre-empting the layer with a
        `WrappedFunction(x -> NNlib.circular_pad(x, N_pad; dims=pad_dims))`

  - `groups`: Expected to be an `Int`. It specifies the number of groups to divide a
              convolution into (set `groups = in_chs` for Depthwise Convolutions). `in_chs`
              and `out_chs` must be divisible by `groups`.
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`.

## Inputs

  - `x`: Data satisfying `ndims(x) == N + 2 && size(x, N - 1) == in_chs`, i.e.
         `size(x) = (I_N, ..., I_1, C_in, N)`

## Returns

  - Output of the convolution `y` of size `(O_N, ..., O_1, C_out, N)` where

```math
O_i = \left\lfloor\frac{I_i + p_i + p_{(i + N) \% |p|} - d_i \times (k_i - 1)}{s_i} + 1\right\rfloor
```

  - Empty `NamedTuple()`

## Parameters

  - `weight`: Convolution kernel
  - `bias`: Bias (present if `use_bias=true`)
"""
@concrete struct Conv <: AbstractLuxLayer
    activation
    in_chs <: IntegerType
    out_chs <: IntegerType
    kernel_size <: Tuple{Vararg{IntegerType}}
    stride <: Tuple{Vararg{IntegerType}}
    pad <: Tuple{Vararg{IntegerType}}
    dilation <: Tuple{Vararg{IntegerType}}
    groups <: IntegerType
    init_weight
    init_bias
    use_bias <: StaticBool
end

function Conv(k::Tuple{Vararg{IntegerType}}, ch::Pair{<:IntegerType, <:IntegerType},
        activation=identity; init_weight=glorot_uniform, init_bias=zeros32,
        stride=1, pad=0, dilation=1, groups=1, use_bias::BoolType=True())
    stride = Utils.expand(Val(length(k)), stride)
    dilation = Utils.expand(Val(length(k)), dilation)
    pad = calc_padding(pad, k, dilation, stride)
    @argcheck allequal(length, (stride, dilation, k))

    return Conv(activation, first(ch), last(ch), k, stride, pad, dilation,
        groups, init_weight, init_bias, static(use_bias))
end

function initialparameters(rng::AbstractRNG, c::Conv)
    weight = init_conv_filter(
        rng, c.kernel_size, c.in_chs => c.out_chs; init=c.init_weight, c.groups)
    has_bias(c) || return (; weight)
    return (; weight, bias=c.init_bias(rng, c.out_chs))
end

function parameterlength(c::Conv)
    return prod(c.kernel_size) * c.in_chs * c.out_chs ÷ c.groups + has_bias(c) * c.out_chs
end

function (c::Conv)(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(c, ps, st, x)
    cdims = DenseConvDims(y, ps.weight; c.stride, padding=c.pad, c.dilation, c.groups)
    bias = safe_getproperty(ps, Val(:bias))
    σ = NNlib.fast_act(c.activation, y)
    return fused_conv_bias_activation(σ, ps.weight, y, bias, cdims), st
end

function Base.show(io::IO, l::Conv)
    print(io, "Conv(", l.kernel_size)
    print(io, ", ", l.in_chs, " => ", l.out_chs)
    l.activation == identity || print(io, ", ", l.activation)
    all(==(0), l.pad) || print(io, ", pad=", PrettyPrinting.tuple_string(l.pad))
    all(==(1), l.stride) || print(io, ", stride=", PrettyPrinting.tuple_string(l.stride))
    all(==(1), l.dilation) ||
        print(io, ", dilation=", PrettyPrinting.tuple_string(l.dilation))
    (l.groups == 1) || print(io, ", groups=", l.groups)
    has_bias(l) || print(io, ", use_bias=false")
    print(io, ")")
end

@doc doc"""
    MaxPool(window::NTuple; pad=0, stride=window)

Max pooling layer, which replaces all pixels in a block of size `window` with the maximum
value.

# Arguments

  - `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling
              `length(window) == 2`

## Keyword Arguments

  - `stride`: Should each be either single integer, or a tuple with `N` integers

  - `pad`: Specifies the number of elements added to the borders of the data array. It can
           be

      + a single integer for equal padding all around,
      + a tuple of `N` integers, to apply the same padding at begin/end of each spatial
        dimension,
      + a tuple of `2*N` integers, for asymmetric padding, or
      + the singleton `SamePad()`, to calculate padding such that
        `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial
        dimension.

# Extended Help

## Inputs

  - `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

## Returns

  - Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where

```math
  O_i = \left\lfloor\frac{I_i + p_i + p_{(i + N) \% |p|} - d_i \times (k_i - 1)}{s_i} + 1\right\rfloor
```

  - Empty `NamedTuple()`

See also [`Conv`](@ref), [`MeanPool`](@ref), [`GlobalMaxPool`](@ref),
[`AdaptiveMaxPool`](@ref)
"""
@concrete struct MaxPool <: AbstractLuxLayer
    k <: Tuple{Vararg{IntegerType}}
    pad <: Tuple{Vararg{IntegerType}}
    stride <: Tuple{Vararg{IntegerType}}
end

function MaxPool(k::Tuple{Vararg{IntegerType}}; pad=0, stride=k)
    stride = Utils.expand(Val(length(k)), stride)
    pad = calc_padding(pad, k, 1, stride)
    @argcheck allequal(length, (stride, k))

    return MaxPool(k, pad, stride)
end

function (m::MaxPool)(x, _, st::NamedTuple)
    return maxpool(x, PoolDims(x, m.k; padding=m.pad, m.stride)), st
end

function Base.show(io::IO, m::MaxPool)
    print(io, "MaxPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", PrettyPrinting.tuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", PrettyPrinting.tuple_string(m.stride))
    return print(io, ")")
end

@doc doc"""
    MeanPool(window::NTuple; pad=0, stride=window)

Mean pooling layer, which replaces all pixels in a block of size `window` with the mean
value.

# Arguments

  - `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling
              `length(window) == 2`

## Keyword Arguments

  - `stride`: Should each be either single integer, or a tuple with `N` integers

  - `pad`: Specifies the number of elements added to the borders of the data array. It can
           be

      + a single integer for equal padding all around,
      + a tuple of `N` integers, to apply the same padding at begin/end of each spatial
        dimension,
      + a tuple of `2*N` integers, for asymmetric padding, or
      + the singleton `SamePad()`, to calculate padding such that
        `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial
        dimension.

# Extended Help

## Inputs

  - `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

## Returns

  - Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where

```math
  O_i = \left\lfloor\frac{I_i + p_i + p_{(i + N) \% |p|} - d_i \times (k_i - 1)}{s_i} + 1\right\rfloor
```

  - Empty `NamedTuple()`

See also [`Conv`](@ref), [`MaxPool`](@ref), [`GlobalMeanPool`](@ref),
[`AdaptiveMeanPool`](@ref)
"""
@concrete struct MeanPool <: AbstractLuxLayer
    k <: Tuple{Vararg{IntegerType}}
    pad <: Tuple{Vararg{IntegerType}}
    stride <: Tuple{Vararg{IntegerType}}
end

function MeanPool(k::Tuple{Vararg{IntegerType}}; pad=0, stride=k)
    stride = Utils.expand(Val(length(k)), stride)
    pad = calc_padding(pad, k, 1, stride)
    @argcheck allequal(length, (stride, k))

    return MeanPool(k, pad, stride)
end

function (m::MeanPool)(x, _, st::NamedTuple)
    return meanpool(x, PoolDims(x, m.k; padding=m.pad, m.stride)), st
end

function Base.show(io::IO, m::MeanPool)
    print(io, "MeanPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", PrettyPrinting.tuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", PrettyPrinting.tuple_string(m.stride))
    return print(io, ")")
end

"""
    Upsample(mode = :nearest; [scale, size])
    Upsample(scale, mode = :nearest)

Upsampling Layer.

## Layer Construction

### Option 1

  - `mode`: Set to `:nearest`, `:linear`, `:bilinear` or `:trilinear`

Exactly one of two keywords must be specified:

  - If `scale` is a number, this applies to all but the last two dimensions (channel and
    batch) of the input.  It may also be a tuple, to control dimensions individually.
  - Alternatively, keyword `size` accepts a tuple, to directly specify the leading
    dimensions of the output.

### Option 2

  - If `scale` is a number, this applies to all but the last two dimensions (channel and
    batch) of the input.  It may also be a tuple, to control dimensions individually.
  - `mode`: Set to `:nearest`, `:bilinear` or `:trilinear`

Currently supported upsampling `mode`s and corresponding NNlib's methods are:

  - `:nearest` -> `NNlib.upsample_nearest`
  - `:bilinear` -> `NNlib.upsample_bilinear`
  - `:trilinear` -> `NNlib.upsample_trilinear`

# Extended Help

## Inputs

  - `x`: For the input dimensions look into the documentation for the corresponding `NNlib`
    function

      + As a rule of thumb, `:nearest` should work with arrays of arbitrary dimensions
      + `:bilinear` works with 4D Arrays
      + `:trilinear` works with 5D Arrays

## Returns

  - Upsampled Input of size `size` or of size `(I_1 x scale[1], ..., I_N x scale[N], C, N)`
  - Empty `NamedTuple()`
"""
@concrete struct Upsample <: AbstractLuxLayer
    scale
    size
    upsample_mode <: StaticSymbol
end

function Upsample(mode::SymbolType=static(:nearest); scale=nothing, size=nothing)
    @argcheck dynamic(mode) in (:nearest, :bilinear, :trilinear)
    if !xor(isnothing(scale), isnothing(size))
        throw(ArgumentError("Either scale or size should be specified (but not both)."))
    end
    return Upsample(scale, size, static(mode))
end

Upsample(scale, mode::SymbolType=static(:nearest)) = Upsample(mode; scale)

function (m::Upsample)(x::AbstractArray, _, st::NamedTuple)
    return lux_upsample_scale_dispatch(m.upsample_mode, x, m.scale), st
end
function (m::Upsample{Nothing})(x::AbstractArray, _, st::NamedTuple)
    return lux_upsample_size_dispatch(m.upsample_mode, x, m.size), st
end

for interp in (:nearest, :bilinear, :trilinear)
    nnlib_interp_func = Symbol(:upsample_, interp)
    @eval begin
        function lux_upsample_scale_dispatch(::StaticSymbol{$(Meta.quot(interp))}, x, scale)
            return $(nnlib_interp_func)(x, scale)
        end
        function lux_upsample_size_dispatch(::StaticSymbol{$(Meta.quot(interp))}, x, size)
            return $(nnlib_interp_func)(x; size)
        end
    end
end

function lux_upsample_scale_dispatch(::StaticSymbol{:nearest}, x, scale::Integer)
    return NNlib.upsample_nearest(x, ntuple(i -> scale, ndims(x) - 2))
end

function Base.show(io::IO, u::Upsample)
    print(io, "Upsample(", u.upsample_mode)
    u.scale !== nothing && print(io, ", scale = $(u.scale)")
    u.size !== nothing && print(io, ", size = $(u.size)")
    print(io, ")")
end

"""
    GlobalMaxPool()

Global Max Pooling layer. Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output,
by performing max pooling on the complete (w,h)-shaped feature maps.

## Inputs

  - `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

## Returns

  - Output of the pooling `y` of size `(1, ..., 1, C, N)`
  - Empty `NamedTuple()`

See also [`MaxPool`](@ref), [`AdaptiveMaxPool`](@ref), [`GlobalMeanPool`](@ref)
"""
struct GlobalMaxPool <: AbstractLuxLayer end

function (g::GlobalMaxPool)(x, _, st::NamedTuple)
    return maxpool(x, PoolDims(x, size(x)[1:(end - 2)])), st
end

"""
    GlobalMeanPool()

Global Mean Pooling layer. Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output,
by performing mean pooling on the complete (w,h)-shaped feature maps.

## Inputs

  - `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

## Returns

  - Output of the pooling `y` of size `(1, ..., 1, C, N)`
  - Empty `NamedTuple()`

See also [`MeanPool`](@ref), [`AdaptiveMeanPool`](@ref), [`GlobalMaxPool`](@ref)
"""
struct GlobalMeanPool <: AbstractLuxLayer end

function (g::GlobalMeanPool)(x, _, st::NamedTuple)
    return meanpool(x, PoolDims(x, size(x)[1:(end - 2)])), st
end

"""
    AdaptiveMaxPool(out::NTuple)

Adaptive Max Pooling layer. Calculates the necessary window size such that its output has
`size(y)[1:N] == out`.

## Arguments

  - `out`: Size of the first `N` dimensions for the output

## Inputs

  - `x`: Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch
    dimensions, after the `N` feature dimensions, where `N = length(out)`.

## Returns

  - Output of size `(out..., C, N)`
  - Empty `NamedTuple()`

See also [`MaxPool`](@ref), [`AdaptiveMeanPool`](@ref).
"""
struct AdaptiveMaxPool{S, O <: Tuple{Vararg{IntegerType}}} <: AbstractLuxLayer
    out::O
    AdaptiveMaxPool(out) = new{length(out) + 2, typeof(out)}(out)
end

function (a::AdaptiveMaxPool{S})(x::AbstractArray{T, S}, _, st::NamedTuple) where {S, T}
    return maxpool(x, compute_adaptive_pooling_dims(x, a.out)), st
end

Base.show(io::IO, a::AdaptiveMaxPool) = print(io, "AdaptiveMaxPool(", a.out, ")")

"""
    AdaptiveMeanPool(out::NTuple)

Adaptive Mean Pooling layer. Calculates the necessary window size such that its output has
`size(y)[1:N] == out`.

## Arguments

  - `out`: Size of the first `N` dimensions for the output

## Inputs

  - `x`: Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch
    dimensions, after the `N` feature dimensions, where `N = length(out)`.

## Returns

  - Output of size `(out..., C, N)`
  - Empty `NamedTuple()`

See also [`MeanPool`](@ref), [`AdaptiveMaxPool`](@ref).
"""
struct AdaptiveMeanPool{S, O <: Tuple{Vararg{IntegerType}}} <: AbstractLuxLayer
    out::O
    AdaptiveMeanPool(out) = new{length(out) + 2, typeof(out)}(out)
end

function (a::AdaptiveMeanPool{S})(x::AbstractArray{T, S}, _, st::NamedTuple) where {S, T}
    return meanpool(x, compute_adaptive_pooling_dims(x, a.out)), st
end

Base.show(io::IO, a::AdaptiveMeanPool) = print(io, "AdaptiveMeanPool(", a.out, ")")

"""
    PixelShuffle(r::Int)

Pixel shuffling layer with upscale factor `r`. Usually used for generating higher
resolution images while upscaling them.

See `NNlib.pixel_shuffle` for more details.

PixelShuffle is not a Layer, rather it returns a [`WrappedFunction`](@ref) with the
function set to `Base.Fix2(pixel_shuffle, r)`

## Arguments

  - `r`: Upscale factor

## Inputs

  - `x`: For 4D-arrays representing N images, the operation converts input
    `size(x) == (W, H, r² x C, N)` to output of size `(r x W, r x H, C, N)`. For
    D-dimensional data, it expects `ndims(x) == D + 2` with channel and batch dimensions, and
    divides the number of channels by `rᴰ`.

## Returns

  - Output of size `(r x W, r x H, C, N)` for 4D-arrays, and `(r x W, r x H, ..., C, N)`
    for D-dimensional data, where `D = ndims(x) - 2`
"""
PixelShuffle(r::IntegerType) = WrappedFunction(Base.Fix2(pixel_shuffle, r))

@doc doc"""
    CrossCor(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
             activation=identity; init_weight=glorot_uniform, init_bias=zeros32, stride=1,
             pad=0, dilation=1, groups=1, use_bias=True())

Cross Correlation layer.

Image data should be stored in WHCN order (width, height, channels, batch). In other words,
a `100 x 100` RGB image would be a `100 x 100 x 3 x 1` array, and a batch of 50 would be a
`100 x 100 x 3 x 50` array. This has `N = 2` spatial dimensions, and needs a kernel size
like `(5, 5)`, a 2-tuple of integers. To take convolutions along `N` feature dimensions,
this layer expects as input an array with `ndims(x) == N + 2`, where
`size(x, N + 1) == in_chs` is the number of input channels, and `size(x, ndims(x))` is the
number of observations in a batch.

## Arguments

  - `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D
         convolutions `length(k) == 2`
  - `in_chs`: Number of input channels
  - `out_chs`: Number of input and output channels
  - `activation`: Activation Function

## Keyword Arguments

  - `init_weight`: Controls the initialization of the weight parameter
  - `init_bias`: Controls the initialization of the bias parameter

  - `stride`: Should each be either single integer, or a tuple with `N` integers
  - `dilation`: Should each be either single integer, or a tuple with `N` integers
  - `groups`: Expected to be an `Int`. It specifies the number of groups to divide a
              convolution into (set `groups = in_chs` for Depthwise Convolutions). `in_chs`
              and `out_chs` must be divisible by `groups`.
  - `pad`: Specifies the number of elements added to the borders of the data array. It can
           be

      + a single integer for equal padding all around,
      + a tuple of `N` integers, to apply the same padding at begin/end of each spatial
        dimension,
      + a tuple of `2*N` integers, for asymmetric padding, or
      + the singleton `SamePad()`, to calculate padding such that
        `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial
        dimension.

  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`.

# Extended Help

## Inputs

  - `x`: Data satisfying `ndims(x) == N + 2 && size(x, N - 1) == in_chs`, i.e.
         `size(x) = (I_N, ..., I_1, C_in, N)`

## Returns

  - Output of the convolution `y` of size `(O_N, ..., O_1, C_out, N)` where

```math
O_i = \left\lfloor\frac{I_i + p_i + p_{(i + N) \% |p|} - d_i \times (k_i - 1)}{s_i} + 1\right\rfloor
```

  - Empty `NamedTuple()`

## Parameters

  - `weight`: Convolution kernel
  - `bias`: Bias (present if `use_bias=true`)
"""
@concrete struct CrossCor <: AbstractLuxLayer
    activation
    in_chs <: IntegerType
    out_chs <: IntegerType
    kernel_size <: Tuple{Vararg{IntegerType}}
    stride <: Tuple{Vararg{IntegerType}}
    pad <: Tuple{Vararg{IntegerType}}
    dilation <: Tuple{Vararg{IntegerType}}
    groups <: IntegerType
    init_weight
    init_bias
    use_bias <: StaticBool
end

function CrossCor(k::Tuple{Vararg{IntegerType}}, ch::Pair{<:IntegerType, <:IntegerType},
        activation=identity; init_weight=glorot_uniform, init_bias=zeros32,
        stride=1, pad=0, dilation=1, groups=1, use_bias::BoolType=True())
    stride = Utils.expand(Val(length(k)), stride)
    dilation = Utils.expand(Val(length(k)), dilation)
    pad = calc_padding(pad, k, dilation, stride)
    @argcheck allequal(length, (stride, dilation, k))

    return CrossCor(activation, first(ch), last(ch), k, stride, pad, dilation,
        groups, init_weight, init_bias, static(use_bias))
end

function initialparameters(rng::AbstractRNG, c::CrossCor)
    weight = init_conv_filter(
        rng, c.kernel_size, c.in_chs => c.out_chs; init=c.init_weight, c.groups)
    has_bias(c) || return (; weight)
    return (; weight, bias=c.init_bias(rng, c.out_chs))
end

function parameterlength(c::CrossCor)
    return prod(c.kernel_size) * c.in_chs * c.out_chs ÷ c.groups + has_bias(c) * c.out_chs
end

function (c::CrossCor)(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(c, ps, st, x)
    cdims = DenseConvDims(
        DenseConvDims(y, ps.weight; c.stride, padding=c.pad, c.dilation, c.groups); F=true)
    bias = safe_getproperty(ps, Val(:bias))
    σ = NNlib.fast_act(c.activation, y)
    return fused_conv_bias_activation(σ, ps.weight, y, bias, cdims), st
end

function Base.show(io::IO, l::CrossCor)
    print(io, "CrossCor(", l.kernel_size)
    print(io, ", ", l.in_chs, " => ", l.out_chs)
    l.activation == identity || print(io, ", ", l.activation)
    all(==(0), l.pad) || print(io, ", pad=", PrettyPrinting.tuple_string(l.pad))
    all(==(1), l.stride) || print(io, ", stride=", PrettyPrinting.tuple_string(l.stride))
    all(==(1), l.dilation) ||
        print(io, ", dilation=", PrettyPrinting.tuple_string(l.dilation))
    (l.groups == 1) || print(io, ", groups=", l.groups)
    has_bias(l) || print(io, ", use_bias=false")
    return print(io, ")")
end

@doc doc"""
    ConvTranspose(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
                  activation=identity; init_weight=glorot_uniform, init_bias=zeros32,
                  stride=1, pad=0, dilation=1, groups=1, use_bias=True())

Standard convolutional transpose layer.

## Arguments

  - `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D
         convolutions `length(k) == 2`
  - `in_chs`: Number of input channels
  - `out_chs`: Number of input and output channels
  - `activation`: Activation Function

## Keyword Arguments

  - `init_weight`: Controls the initialization of the weight parameter
  - `init_bias`: Controls the initialization of the bias parameter

  - `stride`: Should each be either single integer, or a tuple with `N` integers
  - `dilation`: Should each be either single integer, or a tuple with `N` integers
  - `pad`: Specifies the number of elements added to the borders of the data array. It can
           be

      + a single integer for equal padding all around,
      + a tuple of `N` integers, to apply the same padding at begin/end of each spatial
        dimension,
      + a tuple of `2*N` integers, for asymmetric padding, or
      + the singleton `SamePad()`, to calculate padding such that
        `size(output,d) == size(x,d) * stride` (possibly rounded) for each spatial
        dimension.

  - `groups`: Expected to be an `Int`. It specifies the number of groups to divide a
              convolution into (set `groups = in_chs` for Depthwise Convolutions). `in_chs`
              and `out_chs` must be divisible by `groups`.
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`.

# Extended Help

## Inputs

  - `x`: Data satisfying `ndims(x) == N + 2 && size(x, N - 1) == in_chs`, i.e.
         `size(x) = (I_N, ..., I_1, C_in, N)`

## Returns

  - Output of the convolution transpose `y` of size `(O_N, ..., O_1, C_out, N)` where
  - Empty `NamedTuple()`

## Parameters

  - `weight`: Convolution Transpose kernel
  - `bias`: Bias (present if `use_bias=true`)
"""
@concrete struct ConvTranspose <: AbstractLuxLayer
    activation
    in_chs <: IntegerType
    out_chs <: IntegerType
    kernel_size <: Tuple{Vararg{IntegerType}}
    stride <: Tuple{Vararg{IntegerType}}
    pad <: Tuple{Vararg{IntegerType}}
    dilation <: Tuple{Vararg{IntegerType}}
    groups <: IntegerType
    init_weight
    init_bias
    use_bias <: StaticBool
end

function ConvTranspose(
        k::Tuple{Vararg{IntegerType}}, ch::Pair{<:IntegerType, <:IntegerType},
        activation=identity; init_weight=glorot_uniform, init_bias=zeros32,
        stride=1, pad=0, dilation=1, groups=1, use_bias::BoolType=True())
    stride = Utils.expand(Val(length(k)), stride)
    dilation = Utils.expand(Val(length(k)), dilation)
    pad = if pad isa SamePad
        calc_padding(pad, k .- stride .+ 1, dilation, stride)
    else
        calc_padding(pad, k, dilation, stride)
    end
    @argcheck allequal(length, (stride, dilation, k))

    return ConvTranspose(activation, first(ch), last(ch), k, stride, pad, dilation,
        groups, init_weight, init_bias, static(use_bias))
end

function initialparameters(rng::AbstractRNG, c::ConvTranspose)
    weight = init_conv_filter(
        rng, c.kernel_size, c.out_chs => c.in_chs; init=c.init_weight, c.groups)
    has_bias(c) || return (; weight)
    return (; weight, bias=c.init_bias(rng, c.out_chs))
end

function parameterlength(c::ConvTranspose)
    return prod(c.kernel_size) * c.in_chs * c.out_chs ÷ c.groups + has_bias(c) * c.out_chs
end

function (c::ConvTranspose)(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(c, ps, st, x)
    cdims = conv_transpose_dims(y, ps.weight; c.stride, padding=c.pad, c.dilation, c.groups)
    bias = safe_getproperty(ps, Val(:bias))
    σ = NNlib.fast_act(c.activation, y)
    return bias_activation!!(σ, conv_transpose(y, ps.weight, cdims), bias), st
end

function Base.show(io::IO, l::ConvTranspose)
    print(io, "ConvTranspose(", l.kernel_size)
    print(io, ", ", l.in_chs, " => ", l.out_chs)
    l.activation == identity || print(io, ", ", l.activation)
    all(==(0), l.pad) || print(io, ", pad=", PrettyPrinting.tuple_string(l.pad))
    all(==(1), l.stride) || print(io, ", stride=", PrettyPrinting.tuple_string(l.stride))
    all(==(1), l.dilation) ||
        print(io, ", dilation=", PrettyPrinting.tuple_string(l.dilation))
    (l.groups == 1) || print(io, ", groups=", l.groups)
    has_bias(l) || print(io, ", use_bias=false")
    return print(io, ")")
end

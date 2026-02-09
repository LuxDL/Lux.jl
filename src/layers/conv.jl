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
    x::AbstractArray, weight::AbstractArray; padding, stride, dilation, groups, outpad
)
    # Calculate size of "input", from ∇conv_data()'s perspective...
    function calc_dim(xsz, wsz, stride, dilation, pad, outpad)
        return (xsz - 1) * stride + 1 + (wsz - 1) * dilation - pad + outpad
    end
    combined_pad = ntuple(i -> padding[2i - 1] + padding[2i], length(padding) ÷ 2)
    I = map(
        calc_dim,
        size(x)[1:(end - 2)],
        size(weight)[1:(end - 2)],
        stride,
        dilation,
        combined_pad,
        outpad,
    )
    C_in = size(weight)[end - 1] * groups
    C_out = size(weight)[end]
    batch_size = size(x)[end]
    w_size = size(weight)

    size(x)[end - 1] != C_out && throw(
        DimensionMismatch(
            lazy"Expected $(C_out) input channels but got $(size(x)[end - 1]) channels."
        ),
    )

    # Create DenseConvDims() that looks like the corresponding conv()
    return DenseConvDims(
        (I..., C_in, batch_size), w_size; stride, padding, dilation, groups
    )
end

CRC.@non_differentiable conv_transpose_dims(::Any...)

conv_transpose(x, weight, cdims) = LuxLib.Impl.∇conv_data(x, weight, cdims)

function init_conv_weight(
    rng::AbstractRNG,
    init_weight::F,
    filter::NTuple{N,<:IntegerType},
    in_chs::IntegerType,
    out_chs::IntegerType,
    groups,
    σ::A,
) where {F,N,A}
    if init_weight === nothing # Default from PyTorch
        return kaiming_uniform(
            rng,
            Float32,
            filter...,
            in_chs ÷ groups,
            out_chs;
            gain=Utils.calculate_gain(σ, √5.0f0),
        )
    end
    return init_weight(rng, filter..., in_chs ÷ groups, out_chs)
end

function init_conv_bias(
    rng::AbstractRNG,
    init_bias::F,
    filter::NTuple{N,<:IntegerType},
    in_chs::IntegerType,
    out_chs::IntegerType,
    groups,
) where {F,N}
    if init_bias === nothing # Default from PyTorch
        fan_in = prod(filter) * (in_chs ÷ groups)
        bound = inv(sqrt(fan_in))
        y = rand32(rng, out_chs)
        @. y = (y - 0.5f0) * 2 * bound
        return y
    end
    return init_bias(rng, out_chs)
end

construct_crosscor_convdims(::False, cdims::DenseConvDims) = cdims
construct_crosscor_convdims(::True, cdims::DenseConvDims) = DenseConvDims(cdims; F=true)

@doc doc"""
    Conv(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
         activation=identity; init_weight=nothing, init_bias=nothing, stride=1,
         pad=0, dilation=1, groups=1, use_bias=True(), cross_correlation=False())

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
    perform cross-correlation in their convolution layers. Pass `cross_correlation=true` to
    use cross-correlation instead.

## Arguments

  - `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D
         convolutions `length(k) == 2`
  - `in_chs`: Number of input channels
  - `out_chs`: Number of input and output channels
  - `activation`: Activation Function

# Extended Help

## Keyword Arguments

  - `init_weight`: Controls the initialization of the weight parameter. If `nothing`, then
    we use [`kaiming_uniform`](@ref) with gain computed on the basis of the activation
    function (taken from Pytorch
    [`nn.init.calculate_gain`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain)).
  - `init_bias`: Controls the initialization of the bias parameter. If `nothing`, then we
    use uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(fan_in))`.

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
        `WrappedFunction(x -> NNlib.pad_circular(x, N_pad; dims=pad_dims))`

  - `groups`: Expected to be an `Int`. It specifies the number of groups to divide a
              convolution into (set `groups = in_chs` for Depthwise Convolutions). `in_chs`
              and `out_chs` must be divisible by `groups`.
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`.
  - `cross_correlation`: If `true`, perform cross-correlation instead of convolution. Prior
    to `v1`, Lux used to have a `CrossCor` layer which performed cross-correlation. This
    was removed in `v1` in favor of `Conv` with `cross_correlation=true`.

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
    cross_correlation <: StaticBool
end

function Conv(
    k::Tuple{Vararg{IntegerType}},
    ch::Pair{<:IntegerType,<:IntegerType},
    activation=identity;
    init_weight=nothing,
    init_bias=nothing,
    stride=1,
    pad=0,
    dilation=1,
    groups=1,
    use_bias::BoolType=True(),
    cross_correlation::BoolType=False(),
)
    stride = Utils.expand(Val(length(k)), stride)
    dilation = Utils.expand(Val(length(k)), dilation)
    pad = calc_padding(pad, k, dilation, stride)

    ch[1] % groups == 0 ||
        throw(DimensionMismatch("Input channel dimension must be divisible by groups."))
    ch[2] % groups == 0 ||
        throw(DimensionMismatch("Output channel dimension must be divisible by groups."))
    @assert allequal(length, (stride, dilation, k))

    return Conv(
        activation,
        first(ch),
        last(ch),
        k,
        stride,
        pad,
        dilation,
        groups,
        init_weight,
        init_bias,
        static(use_bias),
        static(cross_correlation),
    )
end

function initialparameters(rng::AbstractRNG, c::Conv)
    args = (c.kernel_size, c.in_chs, c.out_chs, c.groups)
    weight = init_conv_weight(rng, c.init_weight, args..., c.activation)
    has_bias(c) || return (; weight)
    return (; weight, bias=init_conv_bias(rng, c.init_bias, args...))
end

function parameterlength(c::Conv)
    return prod(c.kernel_size) * c.in_chs * c.out_chs ÷ c.groups + has_bias(c) * c.out_chs
end

@trace function (c::Conv)(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(c, ps, st, x)
    cdims = construct_crosscor_convdims(
        c.cross_correlation,
        DenseConvDims(y, ps.weight; c.stride, padding=c.pad, c.dilation, c.groups),
    )
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
    known(l.cross_correlation) && print(io, ", cross_correlation=true")
    return print(io, ")")
end

@doc doc"""
    ConvTranspose(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
                  activation=identity; init_weight=glorot_uniform, init_bias=zeros32,
                  stride=1, pad=0, outpad=0, dilation=1, groups=1, use_bias=True(),
                  cross_correlation=False())

Standard convolutional transpose layer.

## Arguments

  - `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D
         convolutions `length(k) == 2`
  - `in_chs`: Number of input channels
  - `out_chs`: Number of input and output channels
  - `activation`: Activation Function

## Keyword Arguments

  - `init_weight`: Controls the initialization of the weight parameter. If `nothing`, then
    we use [`kaiming_uniform`](@ref) with gain computed on the basis of the activation
    function (taken from Pytorch
    [`nn.init.calculate_gain`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain)).
  - `init_bias`: Controls the initialization of the bias parameter. If `nothing`, then we
    use uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(fan_in))`.

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
  - `cross_correlation`: If `true`, perform transposed cross-correlation instead of
    transposed convolution.
 - `outpad`: To converse [`Conv`](@ref) inversability when `stride > 1`, `outpad` can be
    used to increase the size of the output in the desired dimensions. Whereas `pad` is used
    to zero-pad the input, `outpad` only affects the output shape.

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
    outpad <: Tuple{Vararg{IntegerType}}
    dilation <: Tuple{Vararg{IntegerType}}
    groups <: IntegerType
    init_weight
    init_bias
    use_bias <: StaticBool
    cross_correlation <: StaticBool
end

function ConvTranspose(
    k::Tuple{Vararg{IntegerType}},
    ch::Pair{<:IntegerType,<:IntegerType},
    activation=identity;
    init_weight=glorot_uniform,
    init_bias=zeros32,
    stride=1,
    pad=0,
    outpad=0,
    dilation=1,
    groups=1,
    use_bias::BoolType=True(),
    cross_correlation::BoolType=False(),
)
    stride = Utils.expand(Val(length(k)), stride)
    dilation = Utils.expand(Val(length(k)), dilation)
    pad = if pad isa SamePad
        calc_padding(pad, k .- stride .+ 1, dilation, stride)
    else
        calc_padding(pad, k, dilation, stride)
    end
    outpad = Utils.expand(Val(length(k)), outpad)

    ch[2] % groups == 0 ||
        throw(DimensionMismatch("Input channel dimension must be divisible by groups."))
    ch[1] % groups == 0 ||
        throw(DimensionMismatch("Output channel dimension must be divisible by groups."))
    @assert allequal(length, (stride, dilation, k))

    return ConvTranspose(
        activation,
        first(ch),
        last(ch),
        k,
        stride,
        pad,
        outpad,
        dilation,
        groups,
        init_weight,
        init_bias,
        static(use_bias),
        static(cross_correlation),
    )
end

function initialparameters(rng::AbstractRNG, c::ConvTranspose)
    weight = init_conv_weight(
        rng, c.init_weight, c.kernel_size, c.out_chs, c.in_chs, c.groups, c.activation
    )
    has_bias(c) || return (; weight)
    # NOTE: The c.out_chs, c.out_chs is intentional, since it only affects the size of the
    #       bias vector
    return (;
        weight,
        bias=init_conv_bias(
            rng, c.init_bias, c.kernel_size, c.out_chs, c.out_chs, c.groups
        ),
    )
end

function parameterlength(c::ConvTranspose)
    return prod(c.kernel_size) * c.in_chs * c.out_chs ÷ c.groups + has_bias(c) * c.out_chs
end

@trace function (c::ConvTranspose)(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(c, ps, st, x)
    cdims = construct_crosscor_convdims(
        c.cross_correlation,
        conv_transpose_dims(
            y, ps.weight; c.stride, padding=c.pad, c.dilation, c.groups, c.outpad
        ),
    )
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
    all(==(0), l.outpad) || print(io, ", outpad=", PrettyPrinting.tuple_string(l.outpad))
    has_bias(l) || print(io, ", use_bias=false")
    known(l.cross_correlation) && print(io, ", cross_correlation=true")
    return print(io, ")")
end

"""
    Upsample(mode = :nearest; [scale, size, align_corners=false])
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

## Other Keyword Arguments

  - `align_corners`: If `true`, the corner pixels of the input and output tensors are
    aligned, and thus preserving the values at those pixels. This only has effect when mode
    is one of `:bilinear` or `:trilinear`.

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
    align_corners <: Bool
end

function Upsample(
    mode::SymbolType=static(:nearest);
    scale=nothing,
    size=nothing,
    align_corners::Bool=false,
)
    @assert dynamic(mode) in (:nearest, :bilinear, :trilinear)

    if !xor(isnothing(scale), isnothing(size))
        throw(ArgumentError("Either scale or size should be specified (but not both)."))
    end
    return Upsample(scale, size, static(mode), align_corners)
end

Upsample(scale, mode::SymbolType=static(:nearest)) = Upsample(mode; scale)

@trace function (m::Upsample)(x::AbstractArray, _, st::NamedTuple)
    return lux_upsample_scale_dispatch(m.upsample_mode, x, m.scale, m.align_corners), st
end
@trace function (m::Upsample{Nothing})(x::AbstractArray, _, st::NamedTuple)
    return lux_upsample_size_dispatch(m.upsample_mode, x, m.size, m.align_corners), st
end

for interp in (:bilinear, :trilinear)
    nnlib_interp_func = Symbol(:upsample_, interp)
    @eval begin
        function lux_upsample_scale_dispatch(
            ::StaticSymbol{$(Meta.quot(interp))}, x, scale, align_corners
        )
            return $(nnlib_interp_func)(x, scale)
        end
        function lux_upsample_size_dispatch(
            ::StaticSymbol{$(Meta.quot(interp))}, x, size, align_corners
        )
            return $(nnlib_interp_func)(x; size)
        end
    end
end

function lux_upsample_size_dispatch(::StaticSymbol{:nearest}, x, size, _)
    return NNlib.upsample_nearest(x; size)
end
function lux_upsample_scale_dispatch(::StaticSymbol{:nearest}, x, scale, _)
    return NNlib.upsample_nearest(x, scale)
end
function lux_upsample_scale_dispatch(::StaticSymbol{:nearest}, x, scale::Integer, _)
    return NNlib.upsample_nearest(x, ntuple(i -> scale, ndims(x) - 2))
end

function Base.show(io::IO, u::Upsample)
    print(io, "Upsample(", u.upsample_mode)
    u.scale !== nothing && print(io, ", scale = $(u.scale)")
    u.size !== nothing && print(io, ", size = $(u.size)")
    u.align_corners && print(io, ", align_corners = $(u.align_corners)")
    return print(io, ")")
end

"""
    PixelShuffle(r::Int)

Pixel shuffling layer with upscale factor `r`. Usually used for generating higher
resolution images while upscaling them.

See `NNlib.pixel_shuffle` for more details.

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
@concrete struct PixelShuffle <: AbstractLuxWrapperLayer{:layer}
    layer <: AbstractLuxLayer
end

function PixelShuffle(r::IntegerType)
    return PixelShuffle(WrappedFunction(Base.Fix2(pixel_shuffle, r)))
end

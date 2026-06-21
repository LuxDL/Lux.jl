---
url: /dev/api/NN_Primitives/NNlib.md
---
# NNlib {#NNlib-API}

Neural Network Primitives with custom bindings for different accelerator backends in Julia.

::: tip Reexport of `NNlib`

Lux doesn't re-export all of `NNlib` for now. Directly loading `NNlib` is the recommended approach for accessing these functions.

:::

## Attention {#Attention}

```julia
dot_product_attention(query, key, value, [bias]; [fdrop, mask, nheads])
```

Multihead dot product attention used in transformer architectures.

The input arrays must have the first two dimensions given by the number of features and the sequence length, then an arbitrary number of batch dimensions or none.

Returns the attention output array of size `(v_dim, q_len, batch_size...)` and the attention scores of size `(kv_len, q_len, nheads, batch_size...)`.

See also [`dot_product_attention_scores`](/api/NN_Primitives/NNlib#NNlib.dot_product_attention_scores) if you only need the attention scores.

**Arguments**

* `query`: Query array of size `(qk_dim, q_len, batch_size...)`.

* `key`: Key array of size `(qk_dim, kv_len, batch_size...)`.

* `value`: Value array of size `(v_dim, kv_len, batch_size...)`.

* `bias`: Either `nothing` or an array broadcastable to size `(kv_len, q_len, nheads, batch_size)`.         It will be added to the attention scores before applying the softmax. Default `nothing`.

* `fdrop`: A dropout function or layer to be applied on the attention scores right after the softmax.          Default `identity` (no dropout).

* `mask`: Either `nothing` or a boolean array broadcastable to size `(kv_len, q_len, nheads, batch_size)`.         The mask is applied to the attention scores just before the softmax.         See [`make_causal_mask`](/api/NN_Primitives/NNlib#NNlib.make_causal_mask) fore creating causal masks. Default `nothing`.

* `nheads`: Number of heads to split the input arrays into. Default `1`.

**Examples**

```julia
q, k, v = rand(10, 20, 2), rand(10, 30, 2), rand(20, 30, 2)
y, α = dot_product_attention(q, k, v)
```

source

```julia
dot_product_attention_scores(query, key, [bias]; [fdrop, mask])
```

Return the attention scores for the [`dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.dot_product_attention). Input arrays must have dimensions `(num_features ÷ nheads, nheads, sequence_length, batch_size)`.

See [`dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.dot_product_attention) for more details.

source

```julia
make_causal_mask(x, dims=2)
```

Return a boolean square matrix `m` of the same type as `x` and of side `size(x, dims)`. Its elements are set such that `m[i, j] == i ≤ j`.

Can be used to mask the attention scores in [`dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.dot_product_attention).

source

## Softmax {#Softmax}

```julia
softmax(x; dims = 1)
```

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) turns input array `x` into probability distributions that sum to 1 along the dimensions specified by `dims`. It is semantically equivalent to the following:

```julia
softmax(x; dims = 1) = exp.(x) ./ sum(exp.(x), dims = dims)
```

with additional manipulations enhancing numerical stability.

For a matrix input `x` it will by default (`dims = 1`) treat it as a batch of vectors, with each column independent. Keyword `dims = 2` will instead treat rows independently, and so on.

See also [`logsoftmax`](/api/NN_Primitives/NNlib#NNlib.logsoftmax).

**Examples**

```julia
julia> softmax([1, 2, 3])
3-element Vector{Float64}:
 0.09003057317038046
 0.24472847105479764
 0.6652409557748218

julia> softmax([1 2 3; 2 2 2])  # dims=1
2×3 Matrix{Float64}:
 0.268941  0.5  0.731059
 0.731059  0.5  0.268941

julia> softmax([1 2 3; 2 2 2]; dims=2)
2×3 Matrix{Float64}:
 0.0900306  0.244728  0.665241
 0.333333   0.333333  0.333333
```

Note that, when used with Flux.jl, `softmax` must not be passed to layers like `Dense` which accept an activation function. The activation is broadcasted over the result, thus applies to individual numbers. But `softmax` always needs to see the whole column.

```julia
julia> using Flux

julia> x = randn(Float32, 4, 4, 3, 13);

julia> model = Chain(Conv((4, 4), 3 => 8, tanh), Flux.flatten, Dense(8 => 7), softmax);

julia> model(x) |> size
(7, 13)

julia> Dense(4 => 7, softmax)(x)
ERROR: `softmax(x)` called with a number, but it expects an array. 
```

source

```julia
logsoftmax(x; dims = 1)
```

Computes the log of softmax in a more numerically stable way than directly taking `log.(softmax(xs))`. Commonly used in computing cross entropy loss.

It is semantically equivalent to the following:

```julia
logsoftmax(x; dims = 1) = x .- log.(sum(exp.(x), dims = dims))
```

See also [`softmax`](/api/NN_Primitives/NNlib#NNlib.softmax).

source

## Pooling {#Pooling}

```julia
PoolDims(x_size::NTuple{M}, k::Union{NTuple{L, Int}, Int};
         stride=k, padding=0, dilation=1)  where {M, L}
```

Dimensions for a "pooling" operation that can have an arbitrary input size, kernel size, stride, dilation, and channel count.  Used to dispatch onto efficient implementations at compile-time.

source

```julia
maxpool(x, k::NTuple{N, Integer}; pad=0, stride=k)
```

Perform max pool operation with window size `k` on input tensor `x`.

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`

* `pad`: See [`pad_zeros`](/api/NN_Primitives/NNlib#NNlib.pad_zeros) for details.

* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.

source

```julia
meanpool(x, k::NTuple{N, Integer}; pad=0, stride=k)
```

Perform mean pool operation with window size `k` on input tensor `x`.

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`

* `pad`: See [`pad_zeros`](/api/NN_Primitives/NNlib#NNlib.pad_zeros) for details.

* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.

source

```julia
lpnormpool(x, p::Real, k::NTuple{N, Integer}; pad=0, stride=k)
```

Perform Lp pool operation with value of the Lp norm `p` and window size `k` on input tensor `x`, also known as LPPool in pytorch. This pooling operator from [Learned-Norm Pooling for Deep Feedforward and Recurrent Neural Networks](https://arxiv.org/abs/1311.1780).

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`

* `p` is restricted to `0 < p < Inf`.

* `pad`: See [`pad_zeros`](/api/NN_Primitives/NNlib#NNlib.pad_zeros) for details.

* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.

For all elements `x` in a size `k` window, lpnormpool computes `(∑ᵢ xᵢ^p)^(1 / p)` as an element of the output.

Thus `lpnormpool(x, 1, k) ./ prod(k) ≈ meanpool(x, k)` and `lpnormpool(x, 2, k).^2 ./ prod(k) ≈ meanpool(x.^2, k)`.

source

## Padding {#Padding}

```julia
pad_reflect(x, pad::Tuple; [dims])
pad_reflect(x, pad::Int; [dims])
```

Pad the array `x` reflecting its values across the border.

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)` of some length `2n` that specifies the left and right padding size for each of the dimensions in `dims`. If `dims` is not given,  it defaults to the first `n` dimensions.

If `pad` is an integer, it is applied on both sides on every dimension in `dims`. In this case, `dims`  defaults to the first `ndims(x)-2` dimensions  (i.e. excludes the channel and batch dimension).

See also [`pad_repeat`](/api/NN_Primitives/NNlib#NNlib.pad_repeat), [`pad_symmetric`](/api/NN_Primitives/NNlib#NNlib.pad_symmetric), [`pad_circular`](/api/NN_Primitives/NNlib#NNlib.pad_circular), and [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant).

```julia
julia> r = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> pad_reflect(r, (1,2,1,2))
6×6 Matrix{Int64}:
 5  2  5  8  5  2
 4  1  4  7  4  1
 5  2  5  8  5  2
 6  3  6  9  6  3
 5  2  5  8  5  2
 4  1  4  7  4  1
```

source

```julia
pad_symmetric(x, pad::Tuple; [dims])
pad_symmetric(x, pad::Int; [dims])
```

Pad the array `x` reflecting its values symmetrically across the border, i.e. the border values of `x` are present in the padding values, in contrast to [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect).

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)` of some length `2n` that specifies the left and right padding size for each of the dimensions in `dims`. If `dims` is not given,  it defaults to the first `n` dimensions.

If `pad` is an integer, it is applied on both sides on every dimension in `dims`. In this case, `dims`  defaults to the first `ndims(x)-2` dimensions  (i.e. excludes the channel and batch dimension).

See also [`pad_repeat`](/api/NN_Primitives/NNlib#NNlib.pad_repeat), [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect), [`pad_circular`](/api/NN_Primitives/NNlib#NNlib.pad_circular), and [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant).

```julia
julia> r = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> pad_symmetric(r, (1,2,1,2))
6×6 Matrix{Int64}:
 1  1  4  7  7  4
 1  1  4  7  7  4
 2  2  5  8  8  5
 3  3  6  9  9  6
 3  3  6  9  9  6
 2  2  5  8  8  5
```

source

```julia
pad_circular(x, pad::Tuple; [dims])
pad_circular(x, pad::Int; [dims])
```

Pad the array `x` "circularly" across the border by wrapping around values from the opposite side of `x`.

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)` of some length `2n` that specifies the left and right padding size for each of the dimensions in `dims`. If `dims` is not given,  it defaults to the first `n` dimensions.

If `pad` is an integer, it is applied on both sides on every dimension in `dims`. In this case, `dims`  defaults to the first `ndims(x)-2` dimensions  (i.e. excludes the channel and batch dimension).

The pad length on either side in any dimension must not exceed the size of `x` in that dimension, i.e. `pad_circular` is not able to create abitrary sized tilings of `x`.

See also [`pad_repeat`](/api/NN_Primitives/NNlib#NNlib.pad_repeat), [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect), [`pad_symmetric`](/api/NN_Primitives/NNlib#NNlib.pad_symmetric), and [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant).

```julia
julia> r = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> pad_circular(r, (1,2,1,2))
6×6 Matrix{Int64}:
 9  3  6  9  3  6
 7  1  4  7  1  4
 8  2  5  8  2  5
 9  3  6  9  3  6
 7  1  4  7  1  4
 8  2  5  8  2  5
```

source

```julia
pad_repeat(x, pad::Tuple; [dims])
pad_repeat(x, pad::Int; [dims])
```

Pad the array `x` repeating the values on the border.

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)` of some length `2n` that specifies the left and right padding size for each of the dimensions in `dims`. If `dims` is not given,  it defaults to the first `n` dimensions.

If `pad` is an integer, it is applied on both sides on every dimension in `dims`. In this case, `dims`  defaults to the first `ndims(x)-2` dimensions  (i.e. excludes the channel and batch dimension).

See also [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect), [`pad_symmetric`](/api/NN_Primitives/NNlib#NNlib.pad_symmetric), [`pad_circular`](/api/NN_Primitives/NNlib#NNlib.pad_circular), and [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant).

```julia
julia> r = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> pad_repeat(r, (1,2,3,4))
6×10 Matrix{Int64}:
 1  1  1  1  4  7  7  7  7  7
 1  1  1  1  4  7  7  7  7  7
 2  2  2  2  5  8  8  8  8  8
 3  3  3  3  6  9  9  9  9  9
 3  3  3  3  6  9  9  9  9  9
 3  3  3  3  6  9  9  9  9  9
```

source

```julia
pad_constant(x, pad::Tuple, val = 0; [dims = :])
pad_constant(x, pad::Int, val = 0; [dims = :])
```

Pad the array `x` with the constant value `val`.

`pad` can be a tuple of integers. If it is of some length `2 * length(dims)` that specifies the left and right padding size for each of the dimensions in `dims` as `(l1, r1, ..., ln, rn)`.  If supplied with a tuple of length `length(dims)` instead, it applies symmetric padding. If `dims` is not given, it defaults to all dimensions.

For integer `pad` input, it is applied on both sides on every dimension in `dims`.

See also [`pad_zeros`](/api/NN_Primitives/NNlib#NNlib.pad_zeros), [`pad_repeat`](/api/NN_Primitives/NNlib#NNlib.pad_repeat), [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect), [`pad_symmetric`](/api/NN_Primitives/NNlib#NNlib.pad_symmetric), and [`pad_circular`](/api/NN_Primitives/NNlib#NNlib.pad_circular).

```julia
julia> r = reshape(1:4, 2, 2)
2×2 reshape(::UnitRange{Int64}, 2, 2) with eltype Int64:
 1  3
 2  4

julia> pad_constant(r, (1, 2, 3, 4), 8)
5×9 Matrix{Int64}:
 8  8  8  8  8  8  8  8  8
 8  8  8  1  3  8  8  8  8
 8  8  8  2  4  8  8  8  8
 8  8  8  8  8  8  8  8  8
 8  8  8  8  8  8  8  8  8

julia> pad_constant(r, 1, 8)
4×4 Matrix{Int64}:
 8  8  8  8
 8  1  3  8
 8  2  4  8
 8  8  8  8

julia> r = reshape(1:27, 3, 3, 3)
3×3×3 reshape(::UnitRange{Int64}, 3, 3, 3) with eltype Int64:
[:, :, 1] =
 1  4  7
 2  5  8
 3  6  9

[:, :, 2] =
 10  13  16
 11  14  17
 12  15  18

[:, :, 3] =
 19  22  25
 20  23  26
 21  24  27

julia> pad_constant(r, (2,1), dims = 1) # assymetric padding
6×3×3 Array{Int64, 3}:
[:, :, 1] =
 0  0  0
 0  0  0
 1  4  7
 2  5  8
 3  6  9
 0  0  0

[:, :, 2] =
  0   0   0
  0   0   0
 10  13  16
 11  14  17
 12  15  18
  0   0   0

[:, :, 3] =
  0   0   0
  0   0   0
 19  22  25
 20  23  26
 21  24  27
  0   0   0

julia> pad_constant(r, (2,1, 3), dims = (1,2)) # padding must always be either the same length as dims, or double it
ERROR: ArgumentError: Could not parse padding (2, 1, 3) and dims (1, 2)
Stacktrace:
[...]
```

source

```julia
pad_zeros(x, pad::Tuple; [dims])
pad_zeros(x, pad::Int; [dims])
```

Pad the array `x` with zeros. Equivalent to [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant) with the constant equal to 0.

source

## Convolution {#Convolution}

```julia
conv(x, w; stride = 1, pad = 0, dilation = 1, flipped = false, groups = 1)
```

Apply convolution filter `w` to input `x`. `x` and `w` are 3d/4d/5d tensors in 1d/2d/3d convolutions respectively. `x` and `w` may have real or complex element types.

source

```julia
ConvDims
```

Type system-level information about convolution dimensions. Critical for things like `im2col!()` to generate efficient code, and helpful to reduce the number of kwargs getting passed around.

source

```julia
depthwiseconv(x, w; stride=1, pad=0, dilation=1, flipped=false)
```

Depthwise convolution operation with filter `w` on input `x`. `x` and `w` are 3d/4d/5d tensors in 1d/2d/3d convolutions respectively.

source

```julia
DepthwiseConvDims
```

Concrete subclass of `ConvDims` for a depthwise convolution.  Differs primarily due to characterization by `C_in`, `C_mult`, rather than `C_in`, `C_out`.  Useful to be separate from DenseConvDims primarily for channel calculation differences.

source

```julia
DenseConvDims
```

Concrete subclass of `ConvDims` for a normal, dense, conv2d/conv3d.

source

```julia
unfold(x, kernel_size; stride = 1, pad = 0, dilation = 0, flipped = true)
```

Places sliding windows of x into a container tensor of size `(num_windows, window_size, batchsize)`. The window size is determined by the `prod(spatial dims of kernel)*input_channels`. The number of sliding windows will match those of convolution (`conv`) with the same kernel\_size and arguments. Note that by default `conv` flips the spatial dimensions of its kernel (default `flipped=false`), whereas `unfold` does not (default `flipped=true`). Uses `NNlib.im2col!` as backend.

See also [`fold`](/api/NN_Primitives/NNlib#NNlib.fold), the adjoint/transpose operator and a potential inverse of `unfold`.

**Example**

The below example demonstrates that `unfold` uses the same sliding windows as `conv`. In general [`batched_mul`](/api/NN_Primitives/NNlib#NNlib.batched_mul) + `unfold` should not be used to achieve convolution.

```julia
julia> x = reshape([100 2 3 40 5 6 700], 7, 1, 1);  # 1D data, 1 channel, batch of 1

julia> w = reshape([1 0 -1], 3, 1, 1);  # 1D conv kernel of length 3

julia> kws = (pad=1, stride=2, flipped=true);  # use same args for conv and unfold

julia> z = NNlib.unfold(x, size(w); kws...)
4×3×1 Array{Int64, 3}:
[:, :, 1] =
  0  100   2
  2    3  40
 40    5   6
  6  700   0

julia> y1 = conv(x, w; kws...)
4×1×1 Array{Int64, 3}:
[:, :, 1] =
  -2
 -38
  34
   6

julia> y2 = z ⊠ w  # ⊠ (\boxtimes) is NNlib.batched_mul
4×1×1 Array{Int64, 3}:
[:, :, 1] =
  -2
 -38
  34
   6
```

source

```julia
fold(y, output_size, kernel_size; stride = 1, pad = 0, dilation = 0, flipped = true)
```

The adjoint/transpose operator of `unfold`. It accumulates sliding windows from the output of `unfold` into a container tensor of size `output_size`. An inverse to `unfold` may be obtained (in some cases) by using `fold` and accounting for scaling issues with a divisor (see example). Uses `NNlib.col2im!` as backend.

See also [`unfold`](/api/NN_Primitives/NNlib#NNlib.unfold).

**Example**

```julia
julia> x = reshape([100 2 3 40 5 6 700], 7, 1, 1);  # 1D data, 1 channel, batch of 1

julia> y = NNlib.unfold(x, (3,1,1))  # sliding window of size 3
5×3×1 Array{Int64, 3}:
[:, :, 1] =
 100   2    3
   2   3   40
   3  40    5
  40   5    6
   5   6  700

julia> z = NNlib.fold(y, size(x), (3,1,1))  # sum of contributions in y. 100 appears once, 40 three times
7×1×1 Array{Int64, 3}:
[:, :, 1] =
 100
   4
   9
 120
  15
  12
 700

julia> divisor = NNlib.fold(NNlib.unfold(ones(size(x)...), (3,1,1)), size(x), (3,1,1))
7×1×1 Array{Float64, 3}:
[:, :, 1] =
 1.0
 2.0
 3.0
 3.0
 3.0
 2.0
 1.0

julia> z ./ divisor
7×1×1 Array{Float64, 3}:
[:, :, 1] =
 100.0
   2.0
   3.0
  40.0
   5.0
   6.0
 700.0
```

In general, an inverse to `unfold` does not exist if `divisor` contains zeros.

source

## Upsampling {#Upsampling}

```julia
upsample_nearest(x, scale::NTuple{S,Int})
upsample_nearest(x; size::NTuple{S,Int})
```

Upsamples the array `x` by integer multiples along the first `S` dimensions. Subsequent dimensions of `x` are not altered.

Either the `scale` factors or the final output `size` can be specified.

See also [`upsample_bilinear`](/api/NN_Primitives/NNlib#NNlib.upsample_bilinear), for two dimensions of an `N=4` array.

**Example**

```julia
julia> upsample_nearest([1 2 3; 4 5 6], (2, 3))
4×9 Matrix{Int64}:
 1  1  1  2  2  2  3  3  3
 1  1  1  2  2  2  3  3  3
 4  4  4  5  5  5  6  6  6
 4  4  4  5  5  5  6  6  6

julia> ans == upsample_nearest([1 2 3; 4 5 6]; size=(4, 9))  # equivalent
true

julia> upsample_nearest([1 2 3; 4 5 6], (2,))
4×3 Matrix{Int64}:
 1  2  3
 1  2  3
 4  5  6
 4  5  6

julia> ans == upsample_nearest([1 2 3; 4 5 6], size=(4,))
true
```

source

```julia
∇upsample_nearest(Δ::AbstractArray{T,3}, scales::NTuple{S, <:Integer}) where T
```

**Arguments**

* `Δ`: Incoming gradient array, backpropagated from downstream layers

* `scales`: scales by which the image was upsampled in the first place

**Outputs**

* `dx`: Downsampled version of `Δ`

source

```julia
upsample_linear(x::AbstractArray{T,3}, scale::Real; align_corners::Bool = true)
upsample_linear(x::AbstractArray{T,3}; size::Integer, align_corners::Bool = true)
```

Upsamples the first dimension of the array `x` by the upsample provided `scale`, using linear interpolation. As an alternative to using `scale`, the resulting array `size` can be directly specified with a keyword argument.

The size of the output is equal to `(scale*S1, S2, S3)`, where `S1, S2, S3 = size(x)`.

source

```julia
∇upsample_linear(Δ::AbstractArray{T,3}; size::Integer, align_corners::Bool = true) where T
```

**Arguments**

* `Δ`: Incoming gradient array, backpropagated from downstream layers

* `size`: Size of the image upsampled in the first place

**Outputs**

* `dx`: Downsampled version of `Δ`

source

```julia
upsample_bilinear(x::AbstractArray{T,4}, scale::NTuple{2,Real}; align_corners::Bool = true)
upsample_bilinear(x::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = true)
```

Upsamples the first 2 dimensions of the array `x` by the upsample factors stored in `scale`, using bilinear interpolation. As an alternative to using `scale`, the resulting image `size` can be directly specified with a keyword argument.

The size of the output is equal to `(scale[1]*S1, scale[2]*S2, S3, S4)`, where `S1, S2, S3, S4 = size(x)`.

**Examples**

```julia
julia> x = reshape(Float32[1 2 3; 4 5 6], (2,3,1,1))
2×3×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 1.0  2.0  3.0
 4.0  5.0  6.0

julia> upsample_bilinear(x, (2, 3))
4×9×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 1.0  1.25  1.5  1.75  2.0  2.25  2.5  2.75  3.0
 2.0  2.25  2.5  2.75  3.0  3.25  3.5  3.75  4.0
 3.0  3.25  3.5  3.75  4.0  4.25  4.5  4.75  5.0
 4.0  4.25  4.5  4.75  5.0  5.25  5.5  5.75  6.0

julia> ans == upsample_bilinear(x; size=(4, 9))  # specify ouput size instead
true

julia> upsample_bilinear(x, (2.5, 3.5))  # non-integer scaling factors are allowed
5×10×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 1.0   1.22222  1.44444  1.66667  1.88889  …  2.33333  2.55556  2.77778  3.0
 1.75  1.97222  2.19444  2.41667  2.63889     3.08333  3.30556  3.52778  3.75
 2.5   2.72222  2.94444  3.16667  3.38889     3.83333  4.05556  4.27778  4.5
 3.25  3.47222  3.69444  3.91667  4.13889     4.58333  4.80556  5.02778  5.25
 4.0   4.22222  4.44444  4.66667  4.88889     5.33333  5.55556  5.77778  6.0
```

source

```julia
∇upsample_bilinear(Δ::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = true) where T
```

**Arguments**

* `Δ`: Incoming gradient array, backpropagated from downstream layers

* `size`: Lateral (W,H) size of the image upsampled in the first place

**Outputs**

* `dx`: Downsampled version of `Δ`

source

```julia
upsample_trilinear(x::AbstractArray{T,5}, scale::NTuple{3,Real}; align_corners::Bool = true)
upsample_trilinear(x::AbstractArray{T,5}; size::NTuple{3,Integer}, align_corners::Bool = true)
```

Upsamples the first 3 dimensions of the array `x` by the upsample factors stored in `scale`, using trilinear interpolation. As an alternative to using `scale`, the resulting image `size` can be directly specified with a keyword argument.

The size of the output is equal to `(scale[1]*S1, scale[2]*S2, scale[3]*S3, S4, S5)`, where `S1, S2, S3, S4, S5 = size(x)`.

**Examples**

```julia
upsample_trilinear(x, (2, 3, 4))
upsample_trilinear(x; size=(4, 9, 11))  # specify ouput size instead
upsample_trilinear(x, (2.5, 3.5, pi))  # non-integer scaling factors are allowed
```

source

```julia
∇upsample_trilinear(Δ::AbstractArray{T,5}; size::NTuple{3,Integer}, align_corners::Bool = true) where T
```

**Arguments**

* `Δ`: Incoming gradient array, backpropagated from downstream layers

* `size`: Lateral size & depth (W,H,D) of the image upsampled in the first place

**Outputs**

* `dx`: Downsampled version of `Δ`

source

```julia
pixel_shuffle(x, r::Integer)
```

Pixel shuffling operation, upscaling by a factor `r`.

For 4-arrays representing `N` images, the operation converts input `size(x) == (W, H, r^2*C, N)` to output of size `(r*W, r*H, C, N)`. For `D`-dimensional data, it expects `ndims(x) == D+2` with channel and batch dimensions, and divides the number of channels by `r^D`.

Used in super-resolution networks to upsample towards high resolution features. Reference: Shi et. al., "Real-Time Single Image and Video Super-Resolution ...", CVPR 2016, https://arxiv.org/abs/1609.05158

**Examples**

```julia
julia> x = [10i + j + channel/10 for i in 1:2, j in 1:3, channel in 1:4, batch in 1:1]
2×3×4×1 Array{Float64, 4}:
[:, :, 1, 1] =
 11.1  12.1  13.1
 21.1  22.1  23.1

[:, :, 2, 1] =
 11.2  12.2  13.2
 21.2  22.2  23.2

[:, :, 3, 1] =
 11.3  12.3  13.3
 21.3  22.3  23.3

[:, :, 4, 1] =
 11.4  12.4  13.4
 21.4  22.4  23.4

julia> pixel_shuffle(x, 2)  # 4 channels used up as 2x upscaling of image dimensions
4×6×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 11.1  11.3  12.1  12.3  13.1  13.3
 11.2  11.4  12.2  12.4  13.2  13.4
 21.1  21.3  22.1  22.3  23.1  23.3
 21.2  21.4  22.2  22.4  23.2  23.4

julia> y = [i + channel/10 for i in 1:3, channel in 1:6, batch in 1:1]
3×6×1 Array{Float64, 3}:
[:, :, 1] =
 1.1  1.2  1.3  1.4  1.5  1.6
 2.1  2.2  2.3  2.4  2.5  2.6
 3.1  3.2  3.3  3.4  3.5  3.6

julia> pixel_shuffle(y, 2)  # 1D image, with 6 channels reduced to 3
6×3×1 Array{Float64, 3}:
[:, :, 1] =
 1.1  1.3  1.5
 1.2  1.4  1.6
 2.1  2.3  2.5
 2.2  2.4  2.6
 3.1  3.3  3.5
 3.2  3.4  3.6
```

source

## Rotation {#Rotation}

Rotate images in the first two dimensions of an array.

```julia
imrotate(arr::AbstractArray{T, 4}, θ; method=:bilinear, rotation_center=size(arr) .÷ 2 .+ 1)
```

Rotates an array in the first two dimensions around the center pixel `rotation_center`.  The default value of `rotation_center` is defined such that there is a integer center pixel for even and odd sized arrays which it is rotated around. For an even sized array of size `(4,4)` this would be `(3,3)`, for an odd array of size `(3,3)` this would be `(2,2)` However, `rotation_center` can be also non-integer numbers if specified.

The angle `θ` is interpreted in radians.

The adjoint is defined with ChainRulesCore.jl. This method also runs with CUDA (and in principle all KernelAbstractions.jl supported backends).

**Keywords**

* `method=:bilinear` for bilinear interpolation or `method=:nearest` for nearest neighbour

* `rotation_center=size(arr) .÷ 2 .+ 1` means there is a real center pixel around it is rotated.

**Examples**

```julia
julia> arr = zeros((4,4,1,1)); arr[2,2,1,1] = 1;

julia> arr
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> NNlib.imrotate(arr, deg2rad(90)) # rotation around (3,3)
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> NNlib.imrotate(arr, deg2rad(90), rotation_center=(2,2))
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> arr = zeros((3,3,1,1)); arr[1,2,1,1] = 1
1

julia> arr
3×3×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  1.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> NNlib.imrotate(arr, deg2rad(45))
3×3×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.207107  0.0
 0.0  0.0       0.207107
 0.0  0.0       0.0

julia> NNlib.imrotate(arr, deg2rad(45), method=:nearest)
3×3×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  1.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```

source

```julia
∇imrotate(dy, arr::AbstractArray{T, 4}, θ; method=:bilinear,
                                           rotation_center=size(arr) .÷ 2 .+ 1)
```

Adjoint for `imrotate`. Gradient only with respect to `arr` and not `θ`.

**Arguments**

* `dy`: input gradient

* `arr`: Input from primal computation

* `θ`: rotation angle in radians

* `method=:bilinear` or `method=:nearest`

* `rotation_center=size(arr) .÷ 2 .+ 1` rotates around a real center pixel for even and odd sized arrays

source

## Batched Operations {#Batched-Operations}

```julia
batched_mul(A, B) -> C
A ⊠ B  # \boxtimes
```

Batched matrix multiplication. Result has `C[:,:,k...] == A[:,:,k...] * B[:,:,k...]` where `k...` represent  any indices in the last dimensions.

If `ndims(A) == ndims(B) == 3` and `size(B,3) == 1` then instead `C[:,:,k] == A[:,:,k] * B[:,:,1]`, and similarly for `A`.

To transpose each matrix, apply `batched_transpose` to the array, or `batched_adjoint` for conjugate-transpose:

```julia
julia> A, B = randn(2,5,17), randn(5,9,17);

julia> A ⊠ B |> size
(2, 9, 17)

julia> batched_adjoint(A) |> size
(5, 2, 17)

julia> batched_mul(A, batched_adjoint(randn(9,5,17))) |> size
(2, 9, 17)

julia> A ⊠ randn(5,9,1) |> size
(2, 9, 17)

julia> batched_transpose(A) == PermutedDimsArray(A, (2,1,3))
true
```

The equivalent `PermutedDimsArray` may be used in place of `batched_transpose`. Other permutations are also handled by BLAS, provided that the batch index `k` is not the first dimension of the underlying array. Thus `PermutedDimsArray(::Array, (1,3,2))` and `PermutedDimsArray(::Array, (3,1,2))` are fine.

However, `A = PermutedDimsArray(::Array, (3,2,1))` is not acceptable to BLAS, since the batch dimension is the contiguous one: `stride(A,3) == 1`. This will be copied, as doing so is faster than `batched_mul_generic!`.

Both this `copy` and `batched_mul_generic!` produce `@debug` messages, and setting for instance `ENV["JULIA_DEBUG"] = NNlib` will display them.

source

```julia
batched_mul(A::Array{T,3}, B::Matrix)
batched_mul(A::Matrix, B::Array{T,3})
A ⊠ B
```

This is always matrix-matrix multiplication, but either `A` or `B` may lack a batch index.

* When `B` is a matrix, result has `C[:,:,k] == A[:,:,k] * B[:,:]` for all `k`.

* When `A` is a matrix, then `C[:,:,k] == A[:,:] * B[:,:,k]`. This can also be done by reshaping and calling `*`, for instance `A ⊡ B` using TensorCore.jl, but is implemented here using `batched_gemm` instead of `gemm`.

```julia
julia> randn(16,8,32) ⊠ randn(8,4) |> size
(16, 4, 32)

julia> randn(16,8,32) ⊠ randn(8,4,1) |> size  # equivalent
(16, 4, 32)

julia> randn(16,8) ⊠ randn(8,4,32) |> size
(16, 4, 32)
```

See also `batched_vec` to regard `B` as a batch of vectors, `A[:,:,k] * B[:,k]`.

source

```julia
batched_mul!(C, A, B) -> C
batched_mul!(C, A, B, α=1, β=0)
```

In-place batched matrix multiplication, equivalent to `mul!(C[:,:,k], A[:,:,k], B[:,:,k], α, β)` for all `k`. If `size(B,3) == 1` then every batch uses `B[:,:,1]` instead.

This will call `batched_gemm!` whenever possible. For real arrays this means that, for `X ∈ [A,B,C]`, either `stride(X,1)==1` or `stride(X,2)==1`, the latter may be caused by `batched_transpose` or by for instance `PermutedDimsArray(::Array, (3,1,2))`. Unlike `batched_mul` this will never make a copy.

For complex arrays, the wrapper made by `batched_adjoint` must be outermost to be seen. In this case the strided accepted by BLAS are more restricted, if `stride(C,1)==1` then only `stride(AorB::BatchedAdjoint,2) == 1` is accepted.

source

```julia
batched_transpose(A::AbstractArray{T,3})
batched_adjoint(A)
```

Equivalent to applying `transpose` or `adjoint` to each matrix `A[:,:,k]`.

These exist to control how `batched_mul` behaves, as it operates on such matrix slices of an array with `ndims(A)==3`.

`PermutedDimsArray(A, (2,1,3))` is equivalent to `batched_transpose(A)`, and is also understood by `batched_mul` (and more widely supported elsewhere).

```julia
BatchedTranspose{T, S} <: AbstractBatchedMatrix{T, 3}
BatchedAdjoint{T, S}
```

Lazy wrappers analogous to `Transpose` and `Adjoint`, returned by `batched_transpose` etc.

source

```julia
batched_transpose(A::AbstractArray{T,3})
batched_adjoint(A)
```

Equivalent to applying `transpose` or `adjoint` to each matrix `A[:,:,k]`.

These exist to control how `batched_mul` behaves, as it operates on such matrix slices of an array with `ndims(A)==3`.

`PermutedDimsArray(A, (2,1,3))` is equivalent to `batched_transpose(A)`, and is also understood by `batched_mul` (and more widely supported elsewhere).

```julia
BatchedTranspose{T, S} <: AbstractBatchedMatrix{T, 3}
BatchedAdjoint{T, S}
```

Lazy wrappers analogous to `Transpose` and `Adjoint`, returned by `batched_transpose` etc.

source

```julia
batched_vec(A::AbstractArray{T,3}, B::AbstractMatrix)
batched_vec(A::AbstractArray{T,3}, b::AbstractVector)
batched_vec(A::AbstractArray, B::AbstractArray)
```

Batched matrix-vector multiplication. For the 3D case: the result has `C[:,:,k] == A[:,:,k] * B[:,k]` for all `k`, or else `C[:,:,k] == A[:,:,k] * b` for `b::Vector`.

For the general N-D case where `ndims(A) == ndims(B) + 1`: the result has `C[:,k...] == A[:,:,k...] * B[:,k...]` for all batch indices `k...`. The batch dimensions must match: `size(A)[3:end] == size(B)[2:end]`.

With the same argument types, `batched_mul(A, B)` would regard `B` as a fixed matrix, not a batch of vectors. Both reshape and then call `batched_mul(::Array{T,3}, ::Array{T,3})`.

```julia
julia> A, B, b = randn(16,8,32), randn(8,32), randn(8);

julia> batched_vec(A,B) |> size
(16, 32)

julia> batched_vec(A,b) |> size
(16, 32)

julia> A4d, B3d = randn(16,8,10,32), randn(8,10,32);  # 4D and 3D arrays

julia> batched_vec(A4d, B3d) |> size
(16, 10, 32)
```

source

## Gather and Scatter {#Gather-and-Scatter}

```julia
NNlib.gather(src, idx) -> dst
```

Reverse operation of [`scatter`](/api/NN_Primitives/NNlib#NNlib.scatter). Gathers data from source `src` and writes it in a destination `dst` according to the index array `idx`. For each `k` in `CartesianIndices(idx)`, assign values to `dst` according to

```julia
dst[:, ... , k] .= src[:, ... , idx[k]...]
```

Notice that if `idx` is a vector containing integers and `src` is a matrix, previous expression simplifies to

```julia
dst[:, k] .= src[:, idx[k]]
```

and `k` will run over `1:length(idx)`.

The elements of `idx` can be integers or integer tuples and may be repeated. A single `src` column can end up being copied into zero, one, or multiple `dst` columns.

See [`gather!`](/api/NN_Primitives/NNlib#NNlib.gather!) for an in-place version.

**Examples**

```julia
julia> NNlib.gather([1,20,300,4000], [2,4,2])
3-element Vector{Int64}:
   20
 4000
   20

julia> NNlib.gather([1 2 3; 4 5 6], [1,3,1,3,1])
2×5 Matrix{Int64}:
 1  3  1  3  1
 4  6  4  6  4
```

source

```julia
gather(src, IJK...)
```

Convert the tuple of integer vectors `IJK` to a tuple of `CartesianIndex` and call `gather` on it: `gather(src, CartesianIndex.(IJK...))`.

**Examples**

```julia
julia> src = reshape([1:15;], 3, 5)
3×5 Matrix{Int64}:
 1  4  7  10  13
 2  5  8  11  14
 3  6  9  12  15

julia> NNlib.gather(src, [1, 2], [2, 4])
2-element Vector{Int64}:
  4
 11
```

source

```julia
NNlib.gather!(dst, src, idx)
```

Reverse operation of [`scatter!`](/api/NN_Primitives/NNlib#NNlib.scatter!). Gathers data from source `src` and writes it in destination `dst` according to the index array `idx`. For each `k` in `CartesianIndices(idx)`, assign values to `dst` according to

```julia
dst[:, ... , k] .= src[:, ... , idx[k]...]
```

Notice that if `idx` is a vector containing integers, and both `dst` and `src` are matrices, previous expression simplifies to

```julia
dst[:, k] .= src[:, idx[k]]
```

and `k` will run over `1:length(idx)`.

The elements of `idx` can be integers or integer tuples and may be repeated. A single `src` column can end up being copied into zero, one, or multiple `dst` columns.

See [`gather`](/api/NN_Primitives/NNlib#NNlib.gather) for an allocating version.

source

```julia
NNlib.scatter(op, src, idx; [init, dstsize])
```

Scatter operation allocating a destination array `dst` and calling `scatter!(op, dst, src, idx)` on it.

* If keyword `init` is provided, it is used to initialize the content of `dst`. Otherwise, the init values is inferred from the reduction operator `op` for some common operators (e.g. `init = 0` for `op = +`).

* If `dstsize` is provided, it will be used to define the size of destination array, otherwise it will be inferred by `src` and `idx`.

See [`scatter!`](/api/NN_Primitives/NNlib#NNlib.scatter!) for full details on how `idx` works.

**Examples**

```julia
julia> NNlib.scatter(+, [10,100,1000], [3,1,2])
3-element Vector{Int64}:
  100
 1000
   10

julia> NNlib.scatter(+, [1 2 3 4; 5 6 7 8], [2,1,1,5])
2×5 Matrix{Int64}:
  5  1  0  0  4
 13  5  0  0  8

julia> NNlib.scatter(*, [10,200,3000], [1,4,2]; init = 10, dstsize = 6)
6-element Vector{Int64}:
   100
 30000
    10
  2000
    10
    10
```

source

```julia
NNlib.scatter!(op, dst, src, idx)
```

Scatter operation, which writes data in `src` into `dst` at locations `idx`. A binary reduction operator `op` is applied during the scatter. For each index `k` in `idx`, accumulates values in `dst` according to

```julia
dst[:, ..., idx[k]...] = (op).(dst[:, ..., idx[k]...], src[:, ..., k...])
```

See also [`scatter`](/api/NN_Primitives/NNlib#NNlib.scatter), [`gather`](/api/NN_Primitives/NNlib#NNlib.gather).

**Arguments**

* `op`: Operations to be applied on `dst` and `src`, e.g. `+`, `-`, `*`, `/`, `max`, `min` and `mean`.

* `dst`: The destination for `src` to aggregate to. This argument will be mutated.

* `src`: The source data for aggregating.

* `idx`: The mapping for aggregation from source (index) to destination (value).        The `idx` array can contain either integers or tuples.

**Examples**

```julia
julia> NNlib.scatter!(+, ones(3), [10,100], [1,3])
3-element Vector{Float64}:
  11.0
   1.0
 101.0

julia> NNlib.scatter!(*, fill(0.5, 2, 4), [1 10; 100 1000], [3,2])
2×4 Matrix{Float64}:
 0.5    5.0   0.5  0.5
 0.5  500.0  50.0  0.5
```

source

## Sampling {#Sampling}

````julia
grid_sample(input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = :zeros)
grid_sample(input::AbstractArray{T, 5}, grid::AbstractArray{T, 4}; padding_mode = :zeros)

Given `input`, compute output by sampling `input` values at pixel
locations from `grid`. Uses bilinear interpolation to calculate output values.

This implementation assumes the extrema (`-1` and `1`) are considered
as referring to the center points of the input’s corner pixels
(i.e. align corners is `true`).

# Arguments

- `input`: Input array in `(W_in, H_in, [D_in,] C, N)` shape.
- `grid`: Input grid in `(2, W_out, H_out, [D_out,] N)` shape.
    Where for each `(W_out, H_out, [D_out,] N)` grid contains `(x, y [,z])`
    coordinates that specify sampling locations normalized by the `input` shape.

    Therefore, `x`, `y` and [`z`] should have values in `[-1, 1]` range.
    For example, `(x = -1, y = -1, [z = -1])` is the left-top[-front] pixel of `input`,
    and `(x = 1, y = 1, [z = 1])` is the right-bottom-back pixel of `input`.

    Out-of-bound values are handled according to the `padding_mode`.
- `padding_mode`: Out-of-bound padding.
    `:zeros` to use `0` for out-of-bound grid locations.
    `:border` to use border values for out-of-bound grid locations.
    Default is `:zeros`.

# Returns

`(W_out, H_out, [D_out,] C, N)` sampled grid from `input`.

# Examples

In the example below, grid contains two out-of-bound sampling locations,
which are handled differently, depending on the `padding_mode`.

```jldoctest
julia> x = reshape(collect(1.0:4.0), (2, 2, 1, 1))
2×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
1.0  3.0
2.0  4.0

julia> grid = Array{Float64}(undef, 2, 3, 2, 1);

julia> grid[:, 1, 1, 1] .= (-3, -1);

julia> grid[:, 2, 1, 1] .= (0, -1);

julia> grid[:, 3, 1, 1] .= (1, -1);

julia> grid[:, 1, 2, 1] .= (-1, 1);

julia> grid[:, 2, 2, 1] .= (0, 1);

julia> grid[:, 3, 2, 1] .= (3, 1);

julia> grid_sample(x, grid; padding_mode=:zeros)
3×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
0.0  3.0
1.5  3.5
2.0  0.0

julia> grid_sample(x, grid; padding_mode=:border)
3×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
1.0  3.0
1.5  3.5
2.0  4.0
```
````

source

```julia
∇grid_sample(Δ::AbstractArray{T, 4}, input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = :zeros) where T
```

**Arguments**

* `Δ`: Input gradient in `(W_out, H_out, C, N)` shape   (same as output of the primal computation).

* `input`: Input from primal computation in `(W_in, H_in, C, N)` shape.

* `grid`: Grid from primal computation in `(2, W_out, H_out, N)` shape.

* `padding_mode`: Out-of-bound padding.   `:zeros` to use `0` for out-of-bound grid locations.   `:border` to use border values for out-of-bound grid locations.   Should be the same as in primal computation.   Default is `:zeros`.

**Returns**

`dinput` (same shape as `input`) and `dgrid` (same shape as `grid`) gradients.

source

## Losses {#Losses}

```julia
ctc_loss(ŷ, y)
```

Computes the connectionist temporal classification loss between `ŷ` and `y`. `ŷ` must be a classes-by-time matrices, i.e., each row represents a class and each column represents a time step. Additionally, the `logsoftmax` function will be applied to `ŷ`, so `ŷ` must be the raw activation values from the neural network and not, for example, the activations after being passed through a `softmax` activation function. `y` must be a 1D array of the labels associated with `ŷ`. The blank label is assumed to be the last label category in `ŷ`, so it is equivalent to `size(ŷ, 1)`. Used for sequence-to-sequence classification problems such as speech recognition and handwriting recognition where the exact time-alignment of the output (e.g., letters) is not needed to solve the problem. See [Graves et al. (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) or [Graves (2012)](https://www.cs.toronto.edu/~graves/preprint.pdf#chapter.7) for mathematical details.

source

## Miscellaneous {#Miscellaneous}

```julia
logsumexp(x; dims = :)
```

Computes `log.(sum(exp.(x); dims))` in a numerically stable way. Without `dims` keyword this returns a scalar.

See also [`logsoftmax`](/api/NN_Primitives/NNlib#NNlib.logsoftmax).

source

```julia
glu(x, dim = 1)
```

The gated linear unit from the ["Language Modeling with Gated Convolutional Networks"](https://arxiv.org/abs/1612.08083) paper.

Calculates `a .* sigmoid(b)`, where `x` is split in half along given dimension `dim` to form `a` and `b`.

source

```julia
@disallow_spawns ex
```

Disallow NNlib to use `@spawn` on divisible workloads. i.e. within `conv` etc.

source

::: tip Tip

`within_gradient` function currently doesn't work for Enzyme. Prefer to use `LuxLib.Utils.within_autodiff` if needed. Though pay heed that this function is not part of the public API.

:::

```julia
within_gradient(x) --> Bool
```

Returns `false` except when used inside a `gradient` call, when it returns `true`. Useful for Flux regularisation layers which behave differently during training and inference.

This should work with any ChainRules-based differentiation package, in which case `x` is ignored. But Tracker.jl overloads `with_gradient(x::TrackedArray)`, thus for widest use you should pass it an array whose gradient is of interest. There is also an overload for ForwardDiff.jl's `Dual` types (and arrays of them).

**Examples**

```julia
julia> using ForwardDiff, Zygote, NNlib

julia> f_good(x) = if NNlib.within_gradient(x)
                     @show 10x
                   else
                     x
                   end;

julia> Zygote.withgradient(f_good, 1.0)
10x = 10.0
(val = 10.0, grad = (10.0,))

julia> ForwardDiff.derivative(f_good, 1.0)
10x = Dual{ForwardDiff.Tag{typeof(f_good), Float64}}(10.0,10.0)
10.0

julia> f_bad(x, y) = if any(NNlib.within_gradient, (x, y))
                       @show x * y
                     else
                       x / y
                     end;

julia> Zygote.withgradient(f_bad, 2.0, 3.0)
(val = 0.6666666666666666, grad = (0.3333333333333333, -0.2222222222222222))

julia> ForwardDiff.derivative(x -> f_bad(x, 3.0), 2.0)
x * y = Dual{ForwardDiff.Tag{var"#9#10", Float64}}(6.0,3.0)
3.0
```

What goes wrong in `f_bad` is that Zygote knows `any` to be non-differentiable, and thus completely ignores its contents. This is not a perfect mechanism, and the only style recommended is precisely that of `f_good` above.

source

::: tip Tip

Use `LuxLib.API.bias_activation!!` or `LuxLib.API.bias_activation` instead of `NNlib.bias_act!`.

:::

```julia
bias_act!(σ, x, b)
```

This is equivalent to `x .= σ.(x .+ b)`, also replacing `sigmoid` & `tanh` with `sigmoid_fast` & `tanh_fast`. It will only overwrite `x` when `x isa StridedArray{<:AbstractFloat}`.

When used within a gradient, it will overwrite only when `σ` has a method of `derivatives_given_output` which does not need the input at all. Such methods are defined by e.g. `@scalar_rule relu(x) Ω > 0` where the derivative contains only `Ω` (the output) not `x`.

::: warning Warning

This is not safe to use if `x` is still needed for the gradient of some other function. Incorrect use will give silently wrong answers. It is intended mainly for Flux layers, in which the previous operation is known to be safe, e.g. `bias_act!(σ, weight * input, bias)` for a `Dense` layer.

:::

source

## Dropout {#Dropout}

::: tip Tip

Use `LuxLib.API.dropout` instead of `NNlib.dropout`.

:::

```julia
dropout([rng], A, p; [dims])
```

Returns an array in which each element of `A` is either replaced with zero, with probability `p`, or else multiplied by `1/(1-p)`.

By default every element is treated independently. With keyword `dims=1`, a choice is made for every value of the 1st index i.e. each row of a matrix is either zero or not.

Optional first argument is the random number generator used.

**Examples**

```julia
julia> dropout(ones(2, 10), 0.2)
2×10 Matrix{Float64}:
 1.25  1.25  0.0   1.25  1.25  1.25  1.25  1.25  1.25  1.25
 1.25  1.25  1.25  0.0   1.25  1.25  0.0   1.25  1.25  1.25

julia> mean(dropout(ones(10^4, 5), 0.2), dims=1)
1×5 Matrix{Float64}:
 0.998  1.00075  0.99125  0.99575  1.00075

julia> dropout(ones(5, 5), 0.7, dims=1)  # whole row the same
5×5 Matrix{Float64}:
 3.33333  3.33333  3.33333  3.33333  3.33333
 0.0      0.0      0.0      0.0      0.0
 0.0      0.0      0.0      0.0      0.0
 3.33333  3.33333  3.33333  3.33333  3.33333
 0.0      0.0      0.0      0.0      0.0

julia> mean(dropout(ones(10^4, 5), 0.3, dims=1), dims=1)
1×5 Matrix{Float64}:
 1.00571  1.00571  1.00571  1.00571  1.00571
```

source

```julia
dropout!(B, A, p; [dims])
```

This does exactly `B .= dropout(A, p; dims)`, or rather, it's the implementation of out-of-place [`dropout`](/api/NN_Primitives/NNlib#NNlib.dropout).

source

## Internal NNlib Functions {#Internal-NNlib-Functions}

These functions are not part of the public API and are subject to change without notice.

```julia
batched_transpose(A::AbstractArray{T,3})
batched_adjoint(A)
```

Equivalent to applying `transpose` or `adjoint` to each matrix `A[:,:,k]`.

These exist to control how `batched_mul` behaves, as it operates on such matrix slices of an array with `ndims(A)==3`.

`PermutedDimsArray(A, (2,1,3))` is equivalent to `batched_transpose(A)`, and is also understood by `batched_mul` (and more widely supported elsewhere).

```julia
BatchedTranspose{T, S} <: AbstractBatchedMatrix{T, 3}
BatchedAdjoint{T, S}
```

Lazy wrappers analogous to `Transpose` and `Adjoint`, returned by `batched_transpose` etc.

source

```julia
∇conv_filter_direct!(dw, x, dy, cdims; alpha=1, beta=0)
```

Calculate the gradient imposed upon `w` in the convolution `y = x * w`.

source

```julia
_check_trivial_rotations!(out, arr, θ, rotation_center)
```

When `θ = 0 || π /2 || π || 3/2 || π` and if `rotation_center`  is in the middle of the array. For an even array of size 4, the rotation\_center would need to be 2.5. For an odd array of size 5, the rotation\_center would need to be 3.

In those cases, rotations are trivial just by reversing or swapping some axes.

source

```julia
NNlib.fast_act(f, [x::AbstractArray])
```

Replaces `f == tanh` with [`tanh_fast`](/api/NN_Primitives/ActivationFunctions#NNlib.tanh_fast), etc.

Takes an optional 2nd argument, so that you can disable this replacement for some array or element types.

source

```julia
spectrogram(waveform;
    pad::Int = 0, n_fft::Int, hop_length::Int, window,
    center::Bool = true, power::Real = 2.0,
    normalized::Bool = false, window_normalized::Bool = false,
)
```

Create a spectrogram or a batch of spectrograms from a raw audio signal.

**Arguments**

* `pad::Int`:   Then amount of padding to apply on both sides.

* `window_normalized::Bool`:   Whether to normalize the waveform by the window’s L2 energy.

* `power::Real`:   Exponent for the magnitude spectrogram (must be ≥ 0)   e.g., `1` for magnitude, `2` for power, etc.   If `0`, complex spectrum is returned instead.

See [`stft`](/api/NN_Primitives/NNlib#NNlib.stft) for other arguments.

**Returns**

Spectrogram in the shape `(T, F, B)`, where `T` is the number of window hops and `F = n_fft ÷ 2 + 1`.

source

```julia
is_strided(A::AbstractArray) -> Bool
```

This generalises `A isa StridedArray` to treat wrappers like `A::PermutedDimsArray`, for which it returns `is_strided(parent(A))`.

It returns `true` for `CuArray`s, and `PermutedDimsArray`s of those.

Other wrappers (defined outside Base, LinearAlgebra) are assumed not to break strided-ness, and hence also return `is_strided(parent(A))`. This correctly handles things like `NamedDimsArray` wihch don't alter indexing. However, it's a little pessimistic in that e.g. a `view` of such a container will return `false`, even in cases where the same `view` of `parent(A)` would be a `StridedArray`.

source

```julia
conv_direct!(y, x, w, cdims; alpha=1, beta=0)
```

Direct convolution implementation; used for debugging, tests, and mixing/matching of strange datatypes within a single convolution.  Uses naive nested for loop implementation and does not attempt to optimize performance.  Rather, this implementation is intended to be maximally understandable and debuggable, to aid in testing other, more performant implementations.  We also explicitly support mixing and matching of strange datatypes, so that if the user really wants to convolve an image of `UInt8`'s with a `Float16` kernel, storing the result in a `Float32` output, there is at least a function call for that madness.

The keyword arguments `alpha` and `beta` control accumulation behavior; this function calculates `y = alpha * x * w + beta * y`, therefore by setting `beta` to a nonzero value, the user is able to accumulate values into a preallocated `y` buffer, or by setting `alpha` to a nonunitary value, an arbitrary gain factor can be applied.

By defaulting `beta` to `false`, we make use of the Bradbury promotion trick to override `NaN`'s that may pre-exist within our output buffer, as `false*NaN == 0.0`, whereas `0.0*NaN == NaN`.  Only set `beta` if you are certain that none of the elements within `y` are `NaN`.

The basic implementation performs 3-dimensional convolution; 1-dimensional and 2- dimensional cases are supported by simply reshaping `y`, `x` and `w`, for which wrapper methods are available.

source

```julia
gemm!()
```

Low-level gemm!() call with pointers, borrowed from Knet.jl

Calculates `C = alpha*op(A)*op(B) + beta*C`, where:

* `transA` and `transB` set `op(X)` to be either `identity()` or `transpose()`

* alpha and beta are scalars

* op(A) is an (M, K) matrix

* op(B) is a (K, N) matrix

* C is an (M, N) matrix.

source

```julia
calc_padding_regions(dims)
```

Padding is a jerk.  A HUGE jerk that tries to sneak a bunch of conditionals and edge cases (quite literally) into our beautiful stencil operations such as convolution, pooling, etc...  The way we deal with this is to, first, deal with everything in 3d, and then define a single padding region helper function that returns the seven regions that all 3d operations must deal with, including the central "unpadded" region where we can run at full bore, not paying any attention to padding.

source

```julia
∇depthwiseconv_data_im2col!(dx, w, dy, cdims, col=similar(dx); alpha=1, beta=0)
```

Depwthwise conv2d backward pass onto the input using im2col and GEMM. See [`conv_im2col!`](/api/NN_Primitives/NNlib#NNlib.conv_im2col!) for explanation of optional parameters.

source

```julia
_prepare_imrotate(arr, θ, rotation_center)
```

Prepate `sin` and `cos`, creates the output array and converts type of `rotation_center` if required.

source

```julia
insert_singleton_spatial_dimension(cdims::ConvDims)
```

When converting a 1d convolution to a 2d, or a 2d to a 3d, we need to insert a singleton spatial dimension at the end of the spatial dimensions.  This does so for a ConvDims.

source

```julia
_fast_broadcast!(f, x, y, z...)
```

This does `x .= f.(x, y, z...)`, but works around an issue with broadcasting that prevents SIMD in such cases. Can perhaps be removed once https://github.com/JuliaLang/julia/issues/43153 is fixed.

Has an `rrule` to avoid mutation within derivatives.

::: warning Warning

Not intended for general use. Uses `@inbounds` but does not check sizes! Assumes that `f` has no derivative!

:::

source

```julia
hann_window(
    window_length::Int, ::Type{T} = Float32; periodic::Bool = true,
) where T <: Real
```

Hann window function (ref: [Window function § Hann and Hamming windows - Wikipedia](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)).

$w\[n] = \frac{1}{2}\[1 - \cos(\frac{2 \pi n}{N - 1})]$

Where $N$ is the window length.

```julia
julia> lineplot(hann_window(100); width=30, height=10)
     ┌──────────────────────────────┐
   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠚⠉⠉⠉⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⠁⠀⠀⠀⠀⠀⠘⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠞⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⢀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢣⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢦⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⢀⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⢀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢇⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢦⠀⠀⠀⠀│
     │⠀⠀⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠣⡀⠀⠀│
   0 │⣀⣀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢤⣀│
     └──────────────────────────────┘
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀100⠀
```

**Arguments:**

* `window_length::Int`: Size of the window.

* `::Type{T}`: Elemet type of the window.

**Keyword Arguments:**

* `periodic::Bool`: If `true` (default), returns a window to be used as   periodic function. If `false`, return a symmetric window.
  Following always holds:

```julia
julia> N = 256;

julia> hann_window(N; periodic=true) ≈ hann_window(N + 1; periodic=false)[1:end - 1]
true

julia> hann_window(N) ≈ hamming_window(N; α=0.5f0, β=0.5f0)
true
```

**Returns:**

Vector of length `window_length` and eltype `T`.

source

```julia
_rng_from_array(x)
```

Return the random number generator most appropriate for `x`: `CUDA.default_rng()` for `CuArray`, else `Random.default_rng()`

source

```julia
∇depthwiseconv_filter_im2col!(dw, w, dy, cdims, col=similar(dw, ∇filter_im2col_dims(cdims));
                              alpha=1, beta=0)
```

Depthwise conv backward pass onto the weights using im2col and GEMM. See [`conv_im2col!`](/api/NN_Primitives/NNlib#NNlib.conv_im2col!) for explanation of optional parameters.

source

```julia
istft(y;
    n_fft::Int, hop_length::Int = n_fft ÷ 4, window = nothing,
    center::Bool = true, normalized::Bool = false,
    return_complex::Bool = false,
    original_length::Union{Nothing, Int} = nothing,
)
```

Inverse Short-time Fourier Transform.

Return the least squares estimation of the original signal

**Arguments:**

* `y`: Input complex array in the `(n_fft, n_frames, B)` shape.   Where `B` is the optional batch dimension.

**Keyword Arguments:**

* `n_fft::Int`: Size of Fourier transform.

* `hop_length::Int`: Distance between neighboring sliding window frames.

* `window`: Window function that was applied to the input of `stft`.   If `nothing` (default), then no window was applied.

* `center::Bool`: Whether input to `stft` was padded on both sides   so that $t$-th frame is centered at time $t \times \text{hop length}$.   Padding is done with `pad_reflect` function.

* `normalized::Bool`: Whether input to `stft` was normalized.

* `return_complex::Bool`: Whether the output should be complex,   or if the input should be assumed to derive from a real signal and window.

* `original_length::Union{Nothing, Int}`: Optional size of the first dimension   of the input to `stft`. Helps restoring the exact `stft` input size.   Otherwise, the array might be a bit shorter.

source

```julia
transpose_swapbatch(x::AbstractArray)
```

Given an AbstractArray, swap its batch and channel axes, as we must during transposed convolution.  We do this to the operands during convolution, and then again to the output once we're done.

source

```julia
transpose_pad(cdims::ConvDims)
```

Transposed convolution can be calculated in terms of typical convolution with some extra padding.  This method computes the padding of the convolution that would result in the transposed convolution of two operands, in essence taking care of that "extra padding". Note that this method should almost always be accompanied by a call that predilates one of the operands.

source

```julia
power_to_db(s; ref::Real = 1f0, amin::Real = 1f-10, top_db::Real = 80f0)
```

Convert a power spectrogram (amplitude squared) to decibel (dB) units.

**Arguments**

* `s`: Input power.

* `ref`: Scalar w.r.t. which the input is scaled.

* `amin`: Minimum threshold for `s`.

* `top_db`: Threshold the output at `top_db` below the peak:   `max.(s_db, maximum(s_db) - top_db)`.

**Returns**

`s_db ~= 10 * log10(s) - 10 * log10(ref)`

source

```julia
col2im!(x, col, cdims, beta=0)
```

Does the inverse of `im2col!()`, converting `col` back into a 3d image, used for backward passes, transposed convolutions, etc...

Note that this method has not been optimized in the same way as `im2col()` has, because it is slightly more complicated due to the more chaotic data access patterns, and I'm not desperate enough yet.

source

```julia
depthwiseconv_im2col!(y, x, w, cdims, col=similar(x); alpha=1, beta=0)
```

Perform a depthwise convolution using im2col and GEMM, store the result in `y`. See [`conv_im2col!`](/api/NN_Primitives/NNlib#NNlib.conv_im2col!) for explanation of optional parameters.

source

```julia
storage_type(A) -> Type
```

Removes all wrappers to return the `Array` or `CuArray` (or whatever) type within.

```julia
julia> view(reshape(ones(10)',2,5),:, 3:4) |> storage_type
Array{Float64,1}

julia> reshape(sparse(rand(10)), 5,2) |> storage_type
SparseVector{Float64,Int64}
```

source

```julia
im2col_dims(c::ConvDims)
```

im2col calculates, for each output pixel, the "convolution" of N kernels where N is the number of output channels, by doing a matrix multiply.  The dimensions of that matrix are given by this function.

Note that because im2col is multithreaded, we need to allocate a separate workspace of memory per-thread; hence the dimensions returned by this will depend on the number of threads Julia is currently running with.

source

```julia
∇depthwiseconv_filter_direct!(dw, x, dy, cdims; alpha=1, beta=0)
```

Calculate the gradient imposed upon `w` in the depthwise convolution `y = x * w`.

source

```julia
reverse_indices(idx)
```

Return the reverse indices of `idx`. The indices of `idx` will be values, and values of `idx` will be index.

**Arguments**

* `idx`: The indices to be reversed. Accepts array or cuarray of integer, tuple or `CartesianIndex`.

source

```julia
∇conv_filter_im2col!(dw, x, dy, cdims, col=similar(dw, ∇filter_im2col_dims(cdims));
                     alpha=1, beta=0)
```

Conv backward pass onto the weights using im2col and GEMM; stores the result in `dw`. See [`conv_im2col!`](/api/NN_Primitives/NNlib#NNlib.conv_im2col!) for explanation of optional parameters.

source

```julia
conv_im2col!(y, x, w, cdims, col=similar(x); alpha=1, beta=0)
```

Perform a convolution using im2col and GEMM, store the result in `y`.  The  kwargs `alpha` and `beta` control accumulation behavior; internally this operation is implemented as a matrix multiply that boils down to `y = alpha * x * w + beta * y`, thus by setting `beta` to a nonzero value, multiple results can be accumulated into `y`, or by setting `alpha` to a nonunitary value, various gain factors can be applied.

Note for the particularly performance-minded, you can provide a pre-allocated `col`, which should eliminate any need for large allocations within this method.

source

```julia
∇conv_data_direct!(dx, dy, w, cdims; alpha=1, beta=0)
```

Calculate the gradient imposed upon `x` in the convolution `y = x * w`.

source

Performs dimensional consistency checks and return the dimensionality of the scattered objects.

source

```julia
∇conv_data_im2col!(dx, w, dy, cdims, col=similar(dx); alpha=1, beta=0)
```

Conv2d backward pass onto the input using im2col and GEMM; stores the result in `dx`. See [`conv_im2col!`](/api/NN_Primitives/NNlib#NNlib.conv_im2col!) for explanation of optional parameters.

source

```julia
storage_typejoin(A, B, C, ...) -> Type
```

Reduces with `Base.promote_typejoin`, in order that this conveys useful information for dispatching to BLAS. It does not tell you what container to allocate:

```julia
julia> storage_typejoin(rand(2), rand(Float32, 2))
Array{T,1} where T

julia> eltype(ans) <: LinearAlgebra.BlasFloat
false

julia> storage_typejoin(rand(2), rand(2,3), rand(2,3,4))
Array{Float64,N} where N
```

source

```julia
add_blanks(z)
```

Adds blanks to the start and end of `z`, and between items in `z`

source

```julia
∇filter_im2col_dims(c::ConvDims)
```

Like [`im2col_dims`](/api/NN_Primitives/NNlib#NNlib.im2col_dims), but saves some memory because multiple (Julia) threads are not required for the filter gradient calculation.

Note: in the future, this may return `Dims{2}` instead of `Dims{3}`.

source

\_bilinear\_helper(yrot, xrot, yrot\_f, xrot\_f, yrot\_int, xrot\_int)

Some helper variables

source

```julia
_triangular_filterbanks(
    freq_points::Vector{Float32}, all_freqs::Vector{Float32})
```

Create triangular filter banks.

**Arguments:**

* `freq_points::Vector{Float32}`: Filter midpoints of size `n_filters`.

* `all_freqs::Vector{Float32}`: Frequency points of size `n_freqs`.

**Returns:**

Array of size `(n_freqs, n_filters)`.

source

```julia
∇depthwiseconv_data_direct!(dx, dy, w, cdims; alpha=1, beta=0)
```

Calculate the gradient imposed upon `x` in the depthwise convolution `y = x * w`. We make use of the fact that a depthwise convolution is equivalent to `C_in` separate normal convolutions between that channel of `x` and the `C_mult` different kernels that get applied to it.  The output of such a convolution is the gradient imposed upon that particular channel of `x`, and so we simply walk through `x`, calculating the gradient for each batch and channel independently.

source

```julia
db_to_power(s_db; ref::Real = 1f0)
```

Inverse of [`power_to_db`](/api/NN_Primitives/NNlib#NNlib.power_to_db).

source

```julia
predilated_size(x_size::Tuple, dilation::Tuple)
```

Calculate the size of a predilated `x` given a particular dilation factor.  This is used within `predilate()` and `transpose_cdims()`.

source

```julia
stft(x;
    n_fft::Int, hop_length::Int = n_fft ÷ 4, window = nothing,
    center::Bool = true, normalized::Bool = false,
)
```

Short-time Fourier transform (STFT).

The STFT computes the Fourier transform of short overlapping windows of the input, giving frequency components of the signal as they change over time.

$Y\[\omega, m] = \sum\_{k = 0}^{N - 1} \text{window}\[k] \text{input}\[m \times \text{hop length} + k] \exp(-j \frac{2 \pi \omega k}{\text{n fft}})$

where $N$ is the window length, $\omega$ is the frequency $0 \le \omega < \text{n fft}$ and $m$ is the index of the sliding window.

**Arguments:**

* `x`: Input, must be either a 1D time sequence (`(L,)` shape)   or a 2D batch of time sequence (`(L, B)` shape).

**Keyword Arguments:**

* `n_fft::Int`: Size of Fourier transform.

* `hop_length::Int`: Distance between neighboring sliding window frames.

* `window`: Optional window function to apply.   Must be 1D vector `0 < length(window) ≤ n_fft`.   If window is shorter than `n_fft`, it is padded with zeros on both sides.   If `nothing` (default), then no window is applied.

* `center::Bool`: Whether to pad input on both sides so that $t$-th frame   is centered at time $t \times \text{hop length}$.   Padding is done with `pad_reflect` function.

* `normalized::Bool`: Whether to return normalized STFT,   i.e. multiplied with $\text{n fft}^{-0.5}$.

**Returns:**

Complex array of shape `(n_fft, n_frames, B)`, where `B` is the optional batch dimension.

source

```julia
hamming_window(
    window_length::Int, ::Type{T} = Float32; periodic::Bool = true,
    α::T = T(0.54), β::T = T(0.46),
) where T <: Real
```

Hamming window function (ref: [Window function § Hann and Hamming windows - Wikipedia](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)). Generalized version of `hann_window`.

$w\[n] = \alpha - \beta \cos(\frac{2 \pi n}{N - 1})$

Where $N$ is the window length.

```julia
julia> lineplot(hamming_window(100); width=30, height=10)
     ┌──────────────────────────────┐
   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠚⠉⠉⠉⠢⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠁⠀⠀⠀⠀⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⡀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⢰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⣠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⡀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡄⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⡰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀│
     │⠀⠀⠀⢀⠴⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢄⠀⠀⠀│
     │⠀⢀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⣀⠀│
   0 │⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉│
     └──────────────────────────────┘
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀100⠀
```

**Arguments:**

* `window_length::Int`: Size of the window.

* `::Type{T}`: Elemet type of the window.

**Keyword Arguments:**

* `periodic::Bool`: If `true` (default), returns a window to be used as   periodic function. If `false`, return a symmetric window.
  Following always holds:

```julia
julia> N = 256;

julia> hamming_window(N; periodic=true) ≈ hamming_window(N + 1; periodic=false)[1:end - 1]
true
```

* `α::Real`: Coefficient α in the equation above.

* `β::Real`: Coefficient β in the equation above.

**Returns:**

Vector of length `window_length` and eltype `T`.

source

```julia
maximum_dims(dims)
```

Given an array of `CartesianIndex{N}` or `NTuple{N,Int}`, returns a tuple containing the maximum of all the 1st entries, all the 2nd entries, and so on up to `N`.

Given an array of integers, returns `(maximum(dims),)`.

(These arguments are what [`scatter`](/api/NN_Primitives/NNlib#NNlib.scatter) understands.)

source

```julia
batched_transpose(A::AbstractArray{T,3})
batched_adjoint(A)
```

Equivalent to applying `transpose` or `adjoint` to each matrix `A[:,:,k]`.

These exist to control how `batched_mul` behaves, as it operates on such matrix slices of an array with `ndims(A)==3`.

`PermutedDimsArray(A, (2,1,3))` is equivalent to `batched_transpose(A)`, and is also understood by `batched_mul` (and more widely supported elsewhere).

```julia
BatchedTranspose{T, S} <: AbstractBatchedMatrix{T, 3}
BatchedAdjoint{T, S}
```

Lazy wrappers analogous to `Transpose` and `Adjoint`, returned by `batched_transpose` etc.

source

```julia
_rotate_coordinates(sinθ, cosθ, i, j, rotation_center, round_or_floor)
```

This rotates the coordinates and either applies round(nearest neighbour) or floor for :bilinear interpolation)

source

```julia
melscale_filterbanks(;
    n_freqs::Int, n_mels::Int, sample_rate::Int,
    fmin::Float32 = 0f0, fmax::Float32 = Float32(sample_rate ÷ 2))
```

Create triangular Mel scale filter banks (ref: [Mel scale - Wikipedia](https://en.wikipedia.org/wiki/Mel_scale)). Each column is a filterbank that highlights its own frequency.

**Arguments:**

* `n_freqs::Int`: Number of frequencies to highlight.

* `n_mels::Int`: Number of mel filterbanks.

* `sample_rate::Int`: Sample rate of the audio waveform.

* `fmin::Float32`: Minimum frequency in Hz.

* `fmax::Float32`: Maximum frequency in Hz.

**Returns:**

Filterbank matrix of shape `(n_freqs, n_mels)` where each column is a filterbank.

```julia
julia> n_mels = 8;

julia> fb = melscale_filterbanks(; n_freqs=200, n_mels, sample_rate=16000);

julia> plot = lineplot(fb[:, 1]);

julia> for i in 2:n_mels
           lineplot!(plot, fb[:, i])
       end

julia> plot
     ┌────────────────────────────────────────┐
   1 │⠀⡀⢸⠀⢸⠀⠀⣧⠀⠀⢸⡄⠀⠀⠀⣷⠀⠀⠀⠀⠀⣷⠀⠀⠀⠀⠀⠀⢀⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⡇⢸⡆⢸⡇⠀⣿⠀⠀⡜⡇⠀⠀⢰⠋⡆⠀⠀⠀⢰⠁⡇⠀⠀⠀⠀⠀⡸⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⣿⢸⡇⡇⡇⢰⠹⡄⠀⡇⢱⠀⠀⢸⠀⢣⠀⠀⠀⡜⠀⢸⡀⠀⠀⠀⢀⠇⠀⠈⡇⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⣿⡇⡇⡇⡇⢸⠀⡇⢀⠇⠸⡀⠀⡇⠀⠸⡀⠀⢀⠇⠀⠀⢇⠀⠀⠀⡸⠀⠀⠀⠸⡄⠀⠀⠀⠀⠀⠀⠀│
     │⢠⢻⡇⡇⡇⢱⢸⠀⢇⢸⠀⠀⡇⢀⠇⠀⠀⡇⠀⢸⠀⠀⠀⠸⡀⠀⢠⠇⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀│
     │⢸⢸⡇⢱⡇⢸⡇⠀⢸⢸⠀⠀⢣⢸⠀⠀⠀⢸⠀⡇⠀⠀⠀⠀⢇⠀⡜⠀⠀⠀⠀⠀⠈⢇⠀⠀⠀⠀⠀⠀│
     │⢸⢸⡇⢸⠀⢸⡇⠀⢸⡇⠀⠀⢸⡎⠀⠀⠀⠈⣶⠁⠀⠀⠀⠀⠸⣤⠃⠀⠀⠀⠀⠀⠀⠘⡆⠀⠀⠀⠀⠀│
     │⢸⠀⡇⢸⠀⠀⡇⠀⠀⡇⠀⠀⠀⡇⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⢱⡀⠀⠀⠀⠀│
     │⢸⢸⡇⢸⠀⢸⡇⠀⢸⡇⠀⠀⢸⢇⠀⠀⠀⢀⠿⡀⠀⠀⠀⠀⢰⠛⡄⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀│
     │⢸⢸⡇⡸⡇⢸⡇⠀⢸⢸⠀⠀⡜⢸⠀⠀⠀⢸⠀⡇⠀⠀⠀⠀⡎⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠘⡆⠀⠀⠀│
     │⢸⢸⡇⡇⡇⡸⢸⠀⡎⢸⠀⠀⡇⠈⡆⠀⠀⡇⠀⢸⠀⠀⠀⢰⠁⠀⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠸⡄⠀⠀│
     │⡇⢸⡇⡇⡇⡇⢸⠀⡇⠈⡆⢰⠁⠀⡇⠀⢰⠁⠀⠈⡆⠀⠀⡎⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀│
     │⡇⢸⢸⡇⡇⡇⠸⣰⠃⠀⡇⡸⠀⠀⢸⠀⡜⠀⠀⠀⢣⠀⢸⠁⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠈⢇⠀│
     │⡇⡇⢸⠇⢸⡇⠀⣿⠀⠀⢣⡇⠀⠀⠸⣄⠇⠀⠀⠀⠸⡀⡇⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡄│
   0 │⣇⣇⣸⣀⣸⣀⣀⣟⣀⣀⣸⣃⣀⣀⣀⣿⣀⣀⣀⣀⣀⣿⣀⣀⣀⣀⣀⣀⣈⣇⣀⣀⣀⣀⣀⣀⣀⣀⣀⣱│
     └────────────────────────────────────────┘
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀200⠀
```

source

```julia
logaddexp(a, b)
```

Adds log-space `a` and `b` such that the result equals `log(exp(a)+exp(b))`

source

```julia
depthwiseconv_direct!(y, x, w, cdims; alpha=1, beta=0)
```

Direct depthwise convolution implementation; used for debugging, tests, and mixing/ matching of strange datatypes within a single convolution.  Uses naive nested for loop implementation and does not attempt to optimize performance.  Rather, this implementation is intended to be maximally understandable and debuggable, to aid in testing other, more performant implementations.  We also explicitly support mixing and matching of strange datatypes, so that if the user really wants to convolve an image of `UInt8`'s with a `Float16` kernel, storing the result in a `Float32` output, there is at least a function call for that madness.

One subtlety about depthwise convolutions; the shape of a depthwise convolutional kernel is `(spatial_dims..., C_mult, C_in)`, so the axis that must match with the number of channels in `x` is the last, not the second-to-last, as in a normal dense convolution.

See the docstring for `conv_direct!()` for more on the optional parameters.

source

```julia
im2col!(col, x, cdims)
```

Converts a 3d image `x` into a matrix `col` for usage with GEMM-calculated convolution. Patches of `x` of size (kernel\_w, kernel\_h, kernel\_d, C\_in) will be extracted and laid out along the rows of `col`, one for each output pixel.  This routine is used by all im2col-based convolutions, just with extra singleton dimensions added in the case of `2d` or `1d` images.

source

```julia
predilate(x, dilation::Tuple)
```

Places elements of `x` within a lattice of zeros, used in expressing a transposed convolution in terms of normal convolution.  Note that while we call this "predilation" for aesthetic reasons, you are typically passing a "stride" value into here.  Yes, transposed convolution is confusing.

source

```julia
safe_div(x, y)
```

Returns `x/y` unless `y==0`, in which case it just returns `x`. (Used internally by `scatter`.)

source

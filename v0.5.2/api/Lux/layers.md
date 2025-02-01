
<a id='Layers'></a>

# Layers




<a id='Index'></a>

## Index

- [`Lux.AdaptiveMaxPool`](#Lux.AdaptiveMaxPool)
- [`Lux.AdaptiveMeanPool`](#Lux.AdaptiveMeanPool)
- [`Lux.AlphaDropout`](#Lux.AlphaDropout)
- [`Lux.BatchNorm`](#Lux.BatchNorm)
- [`Lux.Bilinear`](#Lux.Bilinear)
- [`Lux.BranchLayer`](#Lux.BranchLayer)
- [`Lux.Chain`](#Lux.Chain)
- [`Lux.Conv`](#Lux.Conv)
- [`Lux.ConvTranspose`](#Lux.ConvTranspose)
- [`Lux.CrossCor`](#Lux.CrossCor)
- [`Lux.Dense`](#Lux.Dense)
- [`Lux.Dropout`](#Lux.Dropout)
- [`Lux.Embedding`](#Lux.Embedding)
- [`Lux.FlattenLayer`](#Lux.FlattenLayer)
- [`Lux.GRUCell`](#Lux.GRUCell)
- [`Lux.GlobalMaxPool`](#Lux.GlobalMaxPool)
- [`Lux.GlobalMeanPool`](#Lux.GlobalMeanPool)
- [`Lux.GroupNorm`](#Lux.GroupNorm)
- [`Lux.InstanceNorm`](#Lux.InstanceNorm)
- [`Lux.LSTMCell`](#Lux.LSTMCell)
- [`Lux.LayerNorm`](#Lux.LayerNorm)
- [`Lux.MaxPool`](#Lux.MaxPool)
- [`Lux.Maxout`](#Lux.Maxout)
- [`Lux.MeanPool`](#Lux.MeanPool)
- [`Lux.NoOpLayer`](#Lux.NoOpLayer)
- [`Lux.PairwiseFusion`](#Lux.PairwiseFusion)
- [`Lux.Parallel`](#Lux.Parallel)
- [`Lux.RNNCell`](#Lux.RNNCell)
- [`Lux.Recurrence`](#Lux.Recurrence)
- [`Lux.ReshapeLayer`](#Lux.ReshapeLayer)
- [`Lux.Scale`](#Lux.Scale)
- [`Lux.SelectDim`](#Lux.SelectDim)
- [`Lux.SkipConnection`](#Lux.SkipConnection)
- [`Lux.StatefulRecurrentCell`](#Lux.StatefulRecurrentCell)
- [`Lux.Upsample`](#Lux.Upsample)
- [`Lux.VariationalHiddenDropout`](#Lux.VariationalHiddenDropout)
- [`Lux.WeightNorm`](#Lux.WeightNorm)
- [`Lux.WrappedFunction`](#Lux.WrappedFunction)
- [`Lux.PixelShuffle`](#Lux.PixelShuffle)


<a id='Containers'></a>

## Containers

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.BranchLayer' href='#Lux.BranchLayer'>#</a>&nbsp;<b><u>Lux.BranchLayer</u></b> &mdash; <i>Type</i>.



```julia
BranchLayer(layers...)
BranchLayer(; name=nothing, layers...)
```

Takes an input `x` and passes it through all the `layers` and returns a tuple of the outputs.

**Arguments**

  * Layers can be specified in two formats:

      * A list of `N` Lux layers
      * Specified as `N` keyword arguments.

**Keyword Arguments**

  * `name`: Name of the layer (optional)

**Inputs**

  * `x`: Will be directly passed to each of the `layers`

**Returns**

  * Tuple: `(layer_1(x), layer_2(x), ..., layer_N(x))`  (naming changes if using the kwargs API)
  * Updated state of the `layers`

**Parameters**

  * Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**States**

  * States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

:::tip Comparison with Parallel

This is slightly different from [`Parallel(nothing, layers...)`](layers#Lux.Parallel)

  * If the input is a tuple, `Parallel` will pass each element individually to each layer.
  * `BranchLayer` essentially assumes 1 input comes in and is branched out into `N` outputs.

:::

**Example**

An easy way to replicate an input to an NTuple is to do

```julia
l = BranchLayer(NoOpLayer(), NoOpLayer(), NoOpLayer())
```


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/containers.jl#L167-L224' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Chain' href='#Lux.Chain'>#</a>&nbsp;<b><u>Lux.Chain</u></b> &mdash; <i>Type</i>.



```julia
Chain(layers...; name=nothing, disable_optimizations::Bool = false)
Chain(; layers..., name=nothing, disable_optimizations::Bool = false)
```

Collects multiple layers / functions to be called in sequence on a given input.

**Arguments**

  * Layers can be specified in two formats:

      * A list of `N` Lux layers
      * Specified as `N` keyword arguments.

**Keyword Arguments**

  * `disable_optimizations`: Prevents any structural optimization
  * `name`: Name of the layer (optional)

**Inputs**

Input `x` is passed sequentially to each layer, and must conform to the input requirements of the internal layers.

**Returns**

  * Output after sequentially applying all the layers to `x`
  * Updated model states

**Parameters**

  * Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**States**

  * States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**Optimizations**

Performs a few optimizations to generate reasonable architectures. Can be disabled using keyword argument `disable_optimizations`.

  * All sublayers are recursively optimized.
  * If a function `f` is passed as a layer and it doesn't take 3 inputs, it is converted to a [`WrappedFunction`](layers#Lux.WrappedFunction)(`f`) which takes only one input.
  * If the layer is a Chain, it is flattened.
  * [`NoOpLayer`](layers#Lux.NoOpLayer)s are removed.
  * If there is only 1 layer (left after optimizations), then it is returned without the `Chain` wrapper.
  * If there are no layers (left after optimizations), a [`NoOpLayer`](layers#Lux.NoOpLayer) is returned.

**Miscellaneous Properties**

  * Allows indexing. We can access the `i`th layer using `m[i]`. We can also index using ranges or arrays.

**Example**

```julia
c = Chain(Dense(2, 3, relu), BatchNorm(3), Dense(3, 2))
```


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/containers.jl#L359-L421' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.PairwiseFusion' href='#Lux.PairwiseFusion'>#</a>&nbsp;<b><u>Lux.PairwiseFusion</u></b> &mdash; <i>Type</i>.



```julia
PairwiseFusion(connection, layers...; name=nothing)
PairwiseFusion(connection; name=nothing, layers...)
```

```
x1 → layer1 → y1 ↘
                  connection → layer2 → y2 ↘
              x2 ↗                          connection → y3
                                        x3 ↗
```

**Arguments**

  * `connection`: Takes 2 inputs and combines them
  * `layers`: `AbstractExplicitLayer`s. Layers can be specified in two formats:

      * A list of `N` Lux layers
      * Specified as `N` keyword arguments.

**Keyword Arguments**

  * `name`: Name of the layer (optional)

**Inputs**

Layer behaves differently based on input type:

1. If the input `x` is a tuple of length `N + 1`, then the `layers` must be a tuple of length `N`. The computation is as follows

```julia
y = x[1]
for i in 1:N
    y = connection(x[i + 1], layers[i](y))
end
```

2. Any other kind of input

```julia
y = x
for i in 1:N
    y = connection(x, layers[i](y))
end
```

**Returns**

  * See Inputs section for how the return value is computed
  * Updated model state for all the contained layers

**Parameters**

  * Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**States**

  * States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/containers.jl#L258-L319' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Parallel' href='#Lux.Parallel'>#</a>&nbsp;<b><u>Lux.Parallel</u></b> &mdash; <i>Type</i>.



```julia
Parallel(connection, layers...; name=nothing)
Parallel(connection; name=nothing, layers...)
```

Create a layer which passes an input to each path in `layers`, before reducing the output with `connection`.

**Arguments**

  * `connection`: An `N`-argument function that is called after passing the input through each layer. If `connection = nothing`, we return a tuple `Parallel(nothing, f, g)(x, y) = (f(x), g(y))`
  * Layers can be specified in two formats:

      * A list of `N` Lux layers
      * Specified as `N` keyword arguments.

**Keyword Arguments**

  * `name`: Name of the layer (optional)

**Inputs**

  * `x`: If `x` is not a tuple, then return is computed as `connection([l(x) for l in layers]...)`. Else one is passed to each layer, thus `Parallel(+, f, g)(x, y) = f(x) + g(y)`.

**Returns**

  * See the Inputs section for how the output is computed
  * Updated state of the `layers`

**Parameters**

  * Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**States**

  * States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

See also [`SkipConnection`](layers#Lux.SkipConnection) which is `Parallel` with one identity.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/containers.jl#L82-L126' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.SkipConnection' href='#Lux.SkipConnection'>#</a>&nbsp;<b><u>Lux.SkipConnection</u></b> &mdash; <i>Type</i>.



```julia
SkipConnection(layer, connection; name=nothing)
```

Create a skip connection which consists of a layer or [`Chain`](layers#Lux.Chain) of consecutive layers and a shortcut connection linking the block's input to the output through a user-supplied 2-argument callable. The first argument to the callable will be propagated through the given `layer` while the second is the unchanged, "skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`.

**Arguments**

  * `layer`: Layer or `Chain` of layers to be applied to the input
  * `connection`:

      * A 2-argument function that takes `layer(input)` and the input OR
      * An AbstractExplicitLayer that takes `(layer(input), input)` as input

**Keyword Arguments**

  * `name`: Name of the layer (optional)

**Inputs**

  * `x`: Will be passed directly to `layer`

**Returns**

  * Output of `connection(layer(input), input)`
  * Updated state of `layer`

**Parameters**

  * Parameters of `layer` OR
  * If `connection` is an AbstractExplicitLayer, then NamedTuple with fields `:layers` and `:connection`

**States**

  * States of `layer` OR
  * If `connection` is an AbstractExplicitLayer, then NamedTuple with fields `:layers` and `:connection`

See [`Parallel`](layers#Lux.Parallel) for a more general implementation.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/containers.jl#L1-L46' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Convolutional Layers'></a>

## Convolutional Layers

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Conv' href='#Lux.Conv'>#</a>&nbsp;<b><u>Lux.Conv</u></b> &mdash; <i>Type</i>.



```julia
Conv(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
     activation=identity; init_weight=glorot_uniform, init_bias=zeros32, stride=1,
     pad=0, dilation=1, groups=1, use_bias=true)
```

Standard convolutional layer.

Image data should be stored in WHCN order (width, height, channels, batch). In other words, a `100 x 100` RGB image would be a `100 x 100 x 3 x 1` array, and a batch of 50 would be a `100 x 100 x 3 x 50` array. This has `N = 2` spatial dimensions, and needs a kernel size like `(5, 5)`, a 2-tuple of integers. To take convolutions along `N` feature dimensions, this layer expects as input an array with `ndims(x) == N + 2`, where `size(x, N + 1) == in_chs` is the number of input channels, and `size(x, ndims(x))` is the number of observations in a batch.

:::warning

Frameworks like [`Pytorch`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d) perform cross-correlation in their convolution layers

:::

**Arguments**

  * `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D      convolutions `length(k) == 2`
  * `in_chs`: Number of input channels
  * `out_chs`: Number of input and output channels
  * `activation`: Activation Function

**Keyword Arguments**

  * `init_weight`: Controls the initialization of the weight parameter
  * `init_bias`: Controls the initialization of the bias parameter
  * `stride`: Should each be either single integer, or a tuple with `N` integers
  * `dilation`: Should each be either single integer, or a tuple with `N` integers
  * `pad`: Specifies the number of elements added to the borders of the data array. It can        be

      * a single integer for equal padding all around,
      * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
      * a tuple of `2*N` integers, for asymmetric padding, or
      * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.
  * `groups`: Expected to be an `Int`. It specifies the number of groups to divide a           convolution into (set `groups = in_chs` for Depthwise Convolutions). `in_chs`           and `out_chs` must be divisible by `groups`.
  * `use_bias`: Trainable bias can be disabled entirely by setting this to `false`.
  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`

**Inputs**

  * `x`: Data satisfying `ndims(x) == N + 2 && size(x, N - 1) == in_chs`, i.e.      `size(x) = (I_N, ..., I_1, C_in, N)`

**Returns**

  * Output of the convolution `y` of size `(O_N, ..., O_1, C_out, N)` where

$$
O_i = floor\left(\frac{I_i + pad[i] + pad[(i + N) \% length(pad)] - dilation[i] \times (k[i] - 1)}{stride[i]} + 1\right)
$$

  * Empty `NamedTuple()`

**Parameters**

  * `weight`: Convolution kernel
  * `bias`: Bias (present if `use_bias=true`)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L1' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.ConvTranspose' href='#Lux.ConvTranspose'>#</a>&nbsp;<b><u>Lux.ConvTranspose</u></b> &mdash; <i>Type</i>.



```julia
ConvTranspose(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
              activation=identity; init_weight=glorot_uniform, init_bias=zeros32,
              stride=1, pad=0, dilation=1, groups=1, use_bias=true)
```

Standard convolutional transpose layer.

**Arguments**

  * `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D      convolutions `length(k) == 2`
  * `in_chs`: Number of input channels
  * `out_chs`: Number of input and output channels
  * `activation`: Activation Function

**Keyword Arguments**

  * `init_weight`: Controls the initialization of the weight parameter
  * `init_bias`: Controls the initialization of the bias parameter
  * `stride`: Should each be either single integer, or a tuple with `N` integers
  * `dilation`: Should each be either single integer, or a tuple with `N` integers
  * `pad`: Specifies the number of elements added to the borders of the data array. It can        be

      * a single integer for equal padding all around,
      * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
      * a tuple of `2*N` integers, for asymmetric padding, or
      * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) * stride` (possibly rounded) for each spatial dimension.
  * `groups`: Expected to be an `Int`. It specifies the number of groups to divide a           convolution into (set `groups = in_chs` for Depthwise Convolutions). `in_chs`           and `out_chs` must be divisible by `groups`.
  * `use_bias`: Trainable bias can be disabled entirely by setting this to `false`.
  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`

**Inputs**

  * `x`: Data satisfying `ndims(x) == N + 2 && size(x, N - 1) == in_chs`, i.e.      `size(x) = (I_N, ..., I_1, C_in, N)`

**Returns**

  * Output of the convolution transpose `y` of size `(O_N, ..., O_1, C_out, N)` where
  * Empty `NamedTuple()`

**Parameters**

  * `weight`: Convolution Transpose kernel
  * `bias`: Bias (present if `use_bias=true`)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L647' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.CrossCor' href='#Lux.CrossCor'>#</a>&nbsp;<b><u>Lux.CrossCor</u></b> &mdash; <i>Type</i>.



```julia
CrossCor(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
         activation=identity; init_weight=glorot_uniform, init_bias=zeros32, stride=1,
         pad=0, dilation=1, use_bias=true)
```

Cross Correlation layer.

Image data should be stored in WHCN order (width, height, channels, batch). In other words, a `100 x 100` RGB image would be a `100 x 100 x 3 x 1` array, and a batch of 50 would be a `100 x 100 x 3 x 50` array. This has `N = 2` spatial dimensions, and needs a kernel size like `(5, 5)`, a 2-tuple of integers. To take convolutions along `N` feature dimensions, this layer expects as input an array with `ndims(x) == N + 2`, where `size(x, N + 1) == in_chs` is the number of input channels, and `size(x, ndims(x))` is the number of observations in a batch.

**Arguments**

  * `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D      convolutions `length(k) == 2`
  * `in_chs`: Number of input channels
  * `out_chs`: Number of input and output channels
  * `activation`: Activation Function

**Keyword Arguments**

  * `init_weight`: Controls the initialization of the weight parameter
  * `init_bias`: Controls the initialization of the bias parameter
  * `stride`: Should each be either single integer, or a tuple with `N` integers
  * `dilation`: Should each be either single integer, or a tuple with `N` integers
  * `pad`: Specifies the number of elements added to the borders of the data array. It can        be

      * a single integer for equal padding all around,
      * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
      * a tuple of `2*N` integers, for asymmetric padding, or
      * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.
  * `use_bias`: Trainable bias can be disabled entirely by setting this to `false`.
  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`

**Inputs**

  * `x`: Data satisfying `ndims(x) == N + 2 && size(x, N - 1) == in_chs`, i.e.      `size(x) = (I_N, ..., I_1, C_in, N)`

**Returns**

  * Output of the convolution `y` of size `(O_N, ..., O_1, C_out, N)` where

$$
O_i = floor\left(\frac{I_i + pad[i] + pad[(i + N) \% length(pad)] - dilation[i] \times (k[i] - 1)}{stride[i]} + 1\right)
$$

  * Empty `NamedTuple()`

**Parameters**

  * `weight`: Convolution kernel
  * `bias`: Bias (present if `use_bias=true`)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L516' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Dropout Layers'></a>

## Dropout Layers

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.AlphaDropout' href='#Lux.AlphaDropout'>#</a>&nbsp;<b><u>Lux.AlphaDropout</u></b> &mdash; <i>Type</i>.



```julia
AlphaDropout(p::Real)
```

AlphaDropout layer.

**Arguments**

  * `p`: Probability of Dropout

      * if `p = 0` then [`NoOpLayer`](layers#Lux.NoOpLayer) is returned.
      * if `p = 1` then `WrappedLayer(Base.Fix1(broadcast, zero))` is returned.

**Inputs**

  * `x`: Must be an AbstractArray

**Returns**

  * `x` with dropout mask applied if `training=Val(true)` else just `x`
  * State with updated `rng`

**States**

  * `rng`: Pseudo Random Number Generator
  * `training`: Used to check if training/inference mode

Call [`Lux.testmode`](../LuxCore/index#LuxCore.testmode) to switch to test mode.

See also [`Dropout`](layers#Lux.Dropout), [`VariationalHiddenDropout`](layers#Lux.VariationalHiddenDropout)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/dropout.jl#L1-L30' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Dropout' href='#Lux.Dropout'>#</a>&nbsp;<b><u>Lux.Dropout</u></b> &mdash; <i>Type</i>.



```julia
Dropout(p; dims=:)
```

Dropout layer.

**Arguments**

  * `p`: Probability of Dropout (if `p = 0` then [`NoOpLayer`](layers#Lux.NoOpLayer) is returned)

**Keyword Arguments**

  * To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input (also called 2D dropout).

**Inputs**

  * `x`: Must be an AbstractArray

**Returns**

  * `x` with dropout mask applied if `training=Val(true)` else just `x`
  * State with updated `rng`

**States**

  * `rng`: Pseudo Random Number Generator
  * `training`: Used to check if training/inference mode

Call [`Lux.testmode`](../LuxCore/index#LuxCore.testmode) to switch to test mode.

See also [`AlphaDropout`](layers#Lux.AlphaDropout), [`VariationalHiddenDropout`](layers#Lux.VariationalHiddenDropout)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/dropout.jl#L62-L94' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.VariationalHiddenDropout' href='#Lux.VariationalHiddenDropout'>#</a>&nbsp;<b><u>Lux.VariationalHiddenDropout</u></b> &mdash; <i>Type</i>.



```julia
VariationalHiddenDropout(p; dims=:)
```

VariationalHiddenDropout layer. The only difference from Dropout is that the `mask` is retained until [`Lux.update_state(l, :update_mask, Val(true))`](../LuxCore/index#LuxCore.update_state) is called.

**Arguments**

  * `p`: Probability of Dropout (if `p = 0` then [`NoOpLayer`](layers#Lux.NoOpLayer) is returned)

**Keyword Arguments**

  * To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `VariationalHiddenDropout(p; dims = 3)` will randomly zero out entire channels on WHCN input (also called 2D dropout).

**Inputs**

  * `x`: Must be an AbstractArray

**Returns**

  * `x` with dropout mask applied if `training=Val(true)` else just `x`
  * State with updated `rng`

**States**

  * `rng`: Pseudo Random Number Generator
  * `training`: Used to check if training/inference mode
  * `mask`: Dropout mask. Initilly set to nothing. After every run, contains the mask applied in that call
  * `update_mask`: Stores whether new mask needs to be generated in the current call

Call [`Lux.testmode`](../LuxCore/index#LuxCore.testmode) to switch to test mode.

See also [`AlphaDropout`](layers#Lux.AlphaDropout), [`Dropout`](layers#Lux.Dropout)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/dropout.jl#L123-L159' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Pooling Layers'></a>

## Pooling Layers

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.AdaptiveMaxPool' href='#Lux.AdaptiveMaxPool'>#</a>&nbsp;<b><u>Lux.AdaptiveMaxPool</u></b> &mdash; <i>Type</i>.



```julia
AdaptiveMaxPool(out::NTuple)
```

Adaptive Max Pooling layer. Calculates the necessary window size such that its output has `size(y)[1:N] == out`.

**Arguments**

  * `out`: Size of the first `N` dimensions for the output

**Inputs**

  * `x`: Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

**Returns**

  * Output of size `(out..., C, N)`
  * Empty `NamedTuple()`

See also [`MaxPool`](layers#Lux.MaxPool), [`AdaptiveMeanPool`](layers#Lux.AdaptiveMeanPool).


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L415-L436' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.AdaptiveMeanPool' href='#Lux.AdaptiveMeanPool'>#</a>&nbsp;<b><u>Lux.AdaptiveMeanPool</u></b> &mdash; <i>Type</i>.



```julia
AdaptiveMeanPool(out::NTuple)
```

Adaptive Mean Pooling layer. Calculates the necessary window size such that its output has `size(y)[1:N] == out`.

**Arguments**

  * `out`: Size of the first `N` dimensions for the output

**Inputs**

  * `x`: Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

**Returns**

  * Output of size `(out..., C, N)`
  * Empty `NamedTuple()`

See also [`MeanPool`](layers#Lux.MeanPool), [`AdaptiveMaxPool`](layers#Lux.AdaptiveMaxPool).


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L451-L472' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.GlobalMaxPool' href='#Lux.GlobalMaxPool'>#</a>&nbsp;<b><u>Lux.GlobalMaxPool</u></b> &mdash; <i>Type</i>.



```julia
GlobalMaxPool()
```

Global Mean Pooling layer. Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output, by performing max pooling on the complete (w,h)-shaped feature maps.

**Inputs**

  * `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

  * Output of the pooling `y` of size `(1, ..., 1, C, N)`
  * Empty `NamedTuple()`

See also [`MaxPool`](layers#Lux.MaxPool), [`AdaptiveMaxPool`](layers#Lux.AdaptiveMaxPool), [`GlobalMeanPool`](layers#Lux.GlobalMeanPool)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L373-L389' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.GlobalMeanPool' href='#Lux.GlobalMeanPool'>#</a>&nbsp;<b><u>Lux.GlobalMeanPool</u></b> &mdash; <i>Type</i>.



```julia
GlobalMeanPool()
```

Global Mean Pooling layer. Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output, by performing mean pooling on the complete (w,h)-shaped feature maps.

**Inputs**

  * `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

  * Output of the pooling `y` of size `(1, ..., 1, C, N)`
  * Empty `NamedTuple()`

See also [`MeanPool`](layers#Lux.MeanPool), [`AdaptiveMeanPool`](layers#Lux.AdaptiveMeanPool), [`GlobalMaxPool`](layers#Lux.GlobalMaxPool)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L394-L410' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.MaxPool' href='#Lux.MaxPool'>#</a>&nbsp;<b><u>Lux.MaxPool</u></b> &mdash; <i>Type</i>.



```julia
MaxPool(window::NTuple; pad=0, stride=window)
```

Max pooling layer, which replaces all pixels in a block of size `window` with the maximum value.

**Arguments**

  * `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling           `length(window) == 2`

**Keyword Arguments**

  * `stride`: Should each be either single integer, or a tuple with `N` integers
  * `pad`: Specifies the number of elements added to the borders of the data array. It can        be

      * a single integer for equal padding all around,
      * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
      * a tuple of `2*N` integers, for asymmetric padding, or
      * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.

**Inputs**

  * `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

  * Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where

$$
  O_i = floor\left(\frac{I_i + pad[i] + pad[(i + N) \% length(pad)] - dilation[i] \times (k[i] - 1)}{stride[i]} + 1\right)
$$

  * Empty `NamedTuple()`

See also [`Conv`](layers#Lux.Conv), [`MeanPool`](layers#Lux.MeanPool), [`GlobalMaxPool`](layers#Lux.GlobalMaxPool), [`AdaptiveMaxPool`](layers#Lux.AdaptiveMaxPool)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L146' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.MeanPool' href='#Lux.MeanPool'>#</a>&nbsp;<b><u>Lux.MeanPool</u></b> &mdash; <i>Type</i>.



```julia
MeanPool(window::NTuple; pad=0, stride=window)
```

Mean pooling layer, which replaces all pixels in a block of size `window` with the mean value.

**Arguments**

  * `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling           `length(window) == 2`

**Keyword Arguments**

  * `stride`: Should each be either single integer, or a tuple with `N` integers
  * `pad`: Specifies the number of elements added to the borders of the data array. It can        be

      * a single integer for equal padding all around,
      * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
      * a tuple of `2*N` integers, for asymmetric padding, or
      * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.

**Inputs**

  * `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

  * Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where

$$
  O_i = floor\left(\frac{I_i + pad[i] + pad[(i + N) \% length(pad)] - dilation[i] \times (k[i] - 1)}{stride[i]} + 1\right)
$$

  * Empty `NamedTuple()`

See also [`Conv`](layers#Lux.Conv), [`MaxPool`](layers#Lux.MaxPool), [`GlobalMeanPool`](layers#Lux.GlobalMeanPool), [`AdaptiveMeanPool`](layers#Lux.AdaptiveMeanPool)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L213' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Recurrent Layers'></a>

## Recurrent Layers

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.GRUCell' href='#Lux.GRUCell'>#</a>&nbsp;<b><u>Lux.GRUCell</u></b> &mdash; <i>Type</i>.



```julia
GRUCell((in_dims, out_dims)::Pair{<:Int,<:Int}; use_bias=true, train_state::Bool=false,
        init_weight::Tuple{Function,Function,Function}=(glorot_uniform, glorot_uniform,
                                                        glorot_uniform),
        init_bias::Tuple{Function,Function,Function}=(zeros32, zeros32, zeros32),
        init_state::Function=zeros32)
```

Gated Recurrent Unit (GRU) Cell

$$
\begin{align}
  r &= \sigma(W_{ir} \times x + W_{hr} \times h_{prev} + b_{hr})\\
  z &= \sigma(W_{iz} \times x + W_{hz} \times h_{prev} + b_{hz})\\
  n &= \tanh(W_{in} \times x + b_{in} + r \cdot (W_{hn} \times h_{prev} + b_{hn}))\\
  h_{new} &= (1 - z) \cdot n + z \cdot h_{prev}
\end{align}
$$

**Arguments**

  * `in_dims`: Input Dimension
  * `out_dims`: Output (Hidden State) Dimension
  * `use_bias`: Set to false to deactivate bias
  * `train_state`: Trainable initial hidden state can be activated by setting this to `true`
  * `init_bias`: Initializer for bias. Must be a tuple containing 3 functions
  * `init_weight`: Initializer for weight. Must be a tuple containing 3 functions
  * `init_state`: Initializer for hidden state

**Inputs**

  * Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `false` - Creates a hidden state using `init_state` and proceeds to Case 2.
  * Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `true` - Repeats `hidden_state` from parameters to match the shape of `x`          and proceeds to Case 2.
  * Case 2: Tuple `(x, (h, ))` is provided, then the output and a tuple containing the          updated hidden state is returned.

**Returns**

  * Tuple containing

      * Output $h_{new}$ of shape `(out_dims, batch_size)`
      * Tuple containing new hidden state $h_{new}$
  * Updated model state

**Parameters**

  * `weight_i`: Concatenated Weights to map from input space             $\\left\\\{ W_{ir}, W_{iz}, W_{in} \\right\\\}$.
  * `weight_h`: Concatenated Weights to map from hidden space             $\\left\\\{ W_{hr}, W_{hz}, W_{hn} \\right\\\}$.
  * `bias_i`: Bias vector ($b_{in}$; not present if `use_bias=false`).
  * `bias_h`: Concatenated Bias vector for the hidden space           $\\left\\\{ b_{hr}, b_{hz}, b_{hn} \\right\\\}$ (not present if           `use_bias=false`).
  * `hidden_state`: Initial hidden state vector (not present if `train_state=false`)           $\\left\\\{ b_{hr}, b_{hz}, b_{hn} \\right\\\}$.

**States**

  * `rng`: Controls the randomness (if any) in the initial state generation


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/recurrent.jl#L463' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.LSTMCell' href='#Lux.LSTMCell'>#</a>&nbsp;<b><u>Lux.LSTMCell</u></b> &mdash; <i>Type</i>.



```julia
LSTMCell(in_dims => out_dims; use_bias::Bool=true, train_state::Bool=false,
         train_memory::Bool=false,
         init_weight=(glorot_uniform, glorot_uniform, glorot_uniform, glorot_uniform),
         init_bias=(zeros32, zeros32, ones32, zeros32), init_state=zeros32,
         init_memory=zeros32)
```

Long Short-Term (LSTM) Cell

$$
\begin{align}
  i &= \sigma(W_{ii} \times x + W_{hi} \times h_{prev} + b_{i})\\
  f &= \sigma(W_{if} \times x + W_{hf} \times h_{prev} + b_{f})\\
  g &= tanh(W_{ig} \times x + W_{hg} \times h_{prev} + b_{g})\\
  o &= \sigma(W_{io} \times x + W_{ho} \times h_{prev} + b_{o})\\
  c_{new} &= f \cdot c_{prev} + i \cdot g\\
  h_{new} &= o \cdot tanh(c_{new})
\end{align}
$$

**Arguments**

  * `in_dims`: Input Dimension
  * `out_dims`: Output (Hidden State & Memory) Dimension
  * `use_bias`: Set to false to deactivate bias
  * `train_state`: Trainable initial hidden state can be activated by setting this to `true`
  * `train_memory`: Trainable initial memory can be activated by setting this to `true`
  * `init_bias`: Initializer for bias. Must be a tuple containing 4 functions
  * `init_weight`: Initializer for weight. Must be a tuple containing 4 functions
  * `init_state`: Initializer for hidden state
  * `init_memory`: Initializer for memory

**Inputs**

  * Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `false`, `train_memory` is set to `false` - Creates a hidden state using          `init_state`, hidden memory using `init_memory` and proceeds to Case 2.
  * Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `true`, `train_memory` is set to `false` - Repeats `hidden_state` vector          from the parameters to match the shape of `x`, creates hidden memory using          `init_memory` and proceeds to Case 2.
  * Case 1c: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `false`, `train_memory` is set to `true` - Creates a hidden state using          `init_state`, repeats the memory vector from parameters to match the shape of          `x` and proceeds to Case 2.
  * Case 1d: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `true`, `train_memory` is set to `true` - Repeats the hidden state and          memory vectors from the parameters to match the shape of  `x` and proceeds to          Case 2.
  * Case 2: Tuple `(x, (h, c))` is provided, then the output and a tuple containing the          updated hidden state and memory is returned.

**Returns**

  * Tuple Containing

      * Output $h_{new}$ of shape `(out_dims, batch_size)`
      * Tuple containing new hidden state $h_{new}$ and new memory $c_{new}$
  * Updated model state

**Parameters**

  * `weight_i`: Concatenated Weights to map from input space             $\{ W_{ii}, W_{if}, W_{ig}, W_{io} \}$.
  * `weight_h`: Concatenated Weights to map from hidden space             $\{ W_{hi}, W_{hf}, W_{hg}, W_{ho} \}$
  * `bias`: Bias vector (not present if `use_bias=false`)
  * `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  * `memory`: Initial memory vector (not present if `train_memory=false`)

**States**

  * `rng`: Controls the randomness (if any) in the initial state generation


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/recurrent.jl#L278' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.RNNCell' href='#Lux.RNNCell'>#</a>&nbsp;<b><u>Lux.RNNCell</u></b> &mdash; <i>Type</i>.



```julia
RNNCell(in_dims => out_dims, activation=tanh; bias::Bool=true,
        train_state::Bool=false, init_bias=zeros32, init_weight=glorot_uniform,
        init_state=ones32)
```

An Elman RNNCell cell with `activation` (typically set to `tanh` or `relu`).

$h_{new} = activation(weight_{ih} \times x + weight_{hh} \times h_{prev} + bias)$

**Arguments**

  * `in_dims`: Input Dimension
  * `out_dims`: Output (Hidden State) Dimension
  * `activation`: Activation function
  * `bias`: Set to false to deactivate bias
  * `train_state`: Trainable initial hidden state can be activated by setting this to `true`
  * `init_bias`: Initializer for bias
  * `init_weight`: Initializer for weight
  * `init_state`: Initializer for hidden state

**Inputs**

  * Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `false` - Creates a hidden state using `init_state` and proceeds to Case 2.
  * Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `true` - Repeats `hidden_state` from parameters to match the shape of `x`          and proceeds to Case 2.
  * Case 2: Tuple `(x, (h, ))` is provided, then the output and a tuple containing the updated hidden state is returned.

**Returns**

  * Tuple containing

      * Output $h_{new}$ of shape `(out_dims, batch_size)`
      * Tuple containing new hidden state $h_{new}$
  * Updated model state

**Parameters**

  * `weight_ih`: Maps the input to the hidden state.
  * `weight_hh`: Maps the hidden state to the hidden state.
  * `bias`: Bias vector (not present if `use_bias=false`)
  * `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

**States**

  * `rng`: Controls the randomness (if any) in the initial state generation


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/recurrent.jl#L160' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Recurrence' href='#Lux.Recurrence'>#</a>&nbsp;<b><u>Lux.Recurrence</u></b> &mdash; <i>Type</i>.



```julia
Recurrence(cell;
    ordering::AbstractTimeSeriesDataBatchOrdering=BatchLastIndex(),
    return_sequence::Bool=false)
```

Wraps a recurrent cell (like [`RNNCell`](layers#Lux.RNNCell), [`LSTMCell`](layers#Lux.LSTMCell), [`GRUCell`](layers#Lux.GRUCell)) to automatically operate over a sequence of inputs.

:::warning

This is completely distinct from `Flux.Recur`. It doesn't make the `cell` stateful, rather allows operating on an entire sequence of inputs at once. See [`StatefulRecurrentCell`](layers#Lux.StatefulRecurrentCell) for functionality similar to `Flux.Recur`.

:::

**Arguments**

  * `cell`: A recurrent cell. See [`RNNCell`](layers#Lux.RNNCell), [`LSTMCell`](layers#Lux.LSTMCell), [`GRUCell`](layers#Lux.GRUCell), for how the inputs/outputs of a recurrent cell must be structured.

**Keyword Arguments**

  * `return_sequence`: If `true` returns the entire sequence of outputs, else returns only the last output. Defaults to `false`.
  * `ordering`: The ordering of the batch and time dimensions in the input. Defaults to `BatchLastIndex()`. Alternatively can be set to `TimeLastIndex()`.

**Inputs**

  * If `x` is a

      * Tuple or Vector: Each element is fed to the `cell` sequentially.
      * Array (except a Vector): It is spliced along the penultimate dimension and each slice is fed to the `cell` sequentially.

**Returns**

  * Output of the `cell` for the entire sequence.
  * Update state of the `cell`.

**Parameters**

  * Same as `cell`.

**States**

  * Same as `cell`.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/recurrent.jl#L8-L57' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.StatefulRecurrentCell' href='#Lux.StatefulRecurrentCell'>#</a>&nbsp;<b><u>Lux.StatefulRecurrentCell</u></b> &mdash; <i>Type</i>.



```julia
StatefulRecurrentCell(cell)
```

Wraps a recurrent cell (like [`RNNCell`](layers#Lux.RNNCell), [`LSTMCell`](layers#Lux.LSTMCell), [`GRUCell`](layers#Lux.GRUCell)) and makes it stateful.

:::tip

This is very similar to `Flux.Recur`

:::

To avoid undefined behavior, once the processing of a single sequence of data is complete, update the state with `Lux.update_state(st, :carry, nothing)`.

**Arguments**

  * `cell`: A recurrent cell. See [`RNNCell`](layers#Lux.RNNCell), [`LSTMCell`](layers#Lux.LSTMCell), [`GRUCell`](layers#Lux.GRUCell), for how the inputs/outputs of a recurrent cell must be structured.

**Inputs**

  * Input to the `cell`.

**Returns**

  * Output of the `cell` for the entire sequence.
  * Update state of the `cell` and updated `carry`.

**Parameters**

  * Same as `cell`.

**States**

  * NamedTuple containing:

      * `cell`: Same as `cell`.
      * `carry`: The carry state of the `cell`.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/recurrent.jl#L98-L137' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Linear Layers'></a>

## Linear Layers

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Bilinear' href='#Lux.Bilinear'>#</a>&nbsp;<b><u>Lux.Bilinear</u></b> &mdash; <i>Type</i>.



```julia
Bilinear((in1_dims, in2_dims) => out, activation=identity; init_weight=glorot_uniform,
         init_bias=zeros32, use_bias::Bool=true, allow_fast_activation::Bool=true)
Bilinear(in12_dims => out, activation=identity; init_weight=glorot_uniform,
         init_bias=zeros32, use_bias::Bool=true, allow_fast_activation::Bool=true)
```

Create a fully connected layer between two inputs and an output, and otherwise similar to [`Dense`](layers#Lux.Dense). Its output, given vectors `x` & `y`, is another vector `z` with, for all `i in 1:out`:

`z[i] = activation(x' * W[i, :, :] * y + bias[i])`

If `x` and `y` are matrices, then each column of the output `z = B(x, y)` is of this form, with `B` the Bilinear layer.

**Arguments**

  * `in1_dims`: number of input dimensions of `x`
  * `in2_dims`: number of input dimensions of `y`
  * `in12_dims`: If specified, then `in1_dims = in2_dims = in12_dims`
  * `out`: number of output dimensions
  * `activation`: activation function

**Keyword Arguments**

  * `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in1_dims, in2_dims)`)
  * `init_bias`: initializer for the bias vector (ignored if `use_bias=false`)
  * `use_bias`: Trainable bias can be disabled entirely by setting this to `false`
  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`

**Input**

  * A 2-Tuple containing

      * `x` must be an AbstractArray with `size(x, 1) == in1_dims`
      * `y` must be an AbstractArray with `size(y, 1) == in2_dims`
  * If the input is an AbstractArray, then `x = y`

**Returns**

  * AbstractArray with dimensions `(out_dims, size(x, 2))`
  * Empty `NamedTuple()`

**Parameters**

  * `weight`: Weight Matrix of size `(out_dims, in1_dims, in2_dims)`
  * `bias`: Bias of size `(out_dims, 1)` (present if `use_bias=true`)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/basic.jl#L306-L357' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Dense' href='#Lux.Dense'>#</a>&nbsp;<b><u>Lux.Dense</u></b> &mdash; <i>Type</i>.



```julia
Dense(in_dims => out_dims, activation=identity; init_weight=glorot_uniform,
      init_bias=zeros32, bias::Bool=true)
```

Create a traditional fully connected layer, whose forward pass is given by: `y = activation.(weight * x .+ bias)`

**Arguments**

  * `in_dims`: number of input dimensions
  * `out_dims`: number of output dimensions
  * `activation`: activation function

**Keyword Arguments**

  * `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in_dims)`)
  * `init_bias`: initializer for the bias vector (ignored if `use_bias=false`)
  * `use_bias`: Trainable bias can be disabled entirely by setting this to `false`
  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`

**Input**

  * `x` must be an AbstractArray with `size(x, 1) == in_dims`

**Returns**

  * AbstractArray with dimensions `(out_dims, ...)` where `...` are the dimensions of `x`
  * Empty `NamedTuple()`

**Parameters**

  * `weight`: Weight Matrix of size `(out_dims, in_dims)`
  * `bias`: Bias of size `(out_dims, 1)` (present if `use_bias=true`)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/basic.jl#L124-L160' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Embedding' href='#Lux.Embedding'>#</a>&nbsp;<b><u>Lux.Embedding</u></b> &mdash; <i>Type</i>.



```julia
Embedding(in_dims => out_dims; init_weight=randn32)
```

A lookup table that stores embeddings of dimension `out_dims` for a vocabulary of size `in_dims`.

This layer is often used to store word embeddings and retrieve them using indices.

:::warning

Unlike `Flux.Embedding`, this layer does not support using `OneHotArray` as an input.

:::

**Arguments**

  * `in_dims`: number of input dimensions
  * `out_dims`: number of output dimensions

**Keyword Arguments**

  * `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in_dims)`)

**Input**

  * Integer OR
  * Abstract Vector of Integers OR
  * Abstract Array of Integers

**Returns**

  * Returns the embedding corresponding to each index in the input. For an N dimensional input, an N + 1 dimensional output is returned.
  * Empty `NamedTuple()`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/basic.jl#L438-L473' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Scale' href='#Lux.Scale'>#</a>&nbsp;<b><u>Lux.Scale</u></b> &mdash; <i>Type</i>.



```julia
Scale(dims, activation=identity; init_weight=ones32, init_bias=zeros32, bias::Bool=true)
```

Create a Sparsely Connected Layer with a very specific structure (only Diagonal Elements are non-zero). The forward pass is given by: `y = activation.(weight .* x .+ bias)`

**Arguments**

  * `dims`: size of the learnable scale and bias parameters.
  * `activation`: activation function

**Keyword Arguments**

  * `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in_dims)`)
  * `init_bias`: initializer for the bias vector (ignored if `use_bias=false`)
  * `use_bias`: Trainable bias can be disabled entirely by setting this to `false`
  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`

**Input**

  * `x` must be an Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])` for `k ≤ size(dims)`

**Returns**

  * Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])` for `k ≤ size(dims)`
  * Empty `NamedTuple()`

**Parameters**

  * `weight`: Weight Array of size `(dims...)`
  * `bias`: Bias of size `(dims...)`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/basic.jl#L224-L259' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Misc. Helper Layers'></a>

## Misc. Helper Layers

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.FlattenLayer' href='#Lux.FlattenLayer'>#</a>&nbsp;<b><u>Lux.FlattenLayer</u></b> &mdash; <i>Type</i>.



```julia
FlattenLayer()
```

Flattens the passed array into a matrix.

**Inputs**

  * `x`: AbstractArray

**Returns**

  * AbstractMatrix of size `(:, size(x, ndims(x)))`
  * Empty `NamedTuple()`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/basic.jl#L31-L44' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Maxout' href='#Lux.Maxout'>#</a>&nbsp;<b><u>Lux.Maxout</u></b> &mdash; <i>Type</i>.



```julia
Maxout(layers...)
Maxout(; layers...)
Maxout(f::Function, n_alts::Int)
```

This contains a number of internal layers, each of which receives the same input. Its output is the elementwise maximum of the the internal layers' outputs.

Maxout over linear dense layers satisfies the univeral approximation theorem. See [1].

See also [`Parallel`](layers#Lux.Parallel) to reduce with other operators.

**Arguments**

  * Layers can be specified in three formats:

      * A list of `N` Lux layers
      * Specified as `N` keyword arguments.
      * A no argument function `f` and an integer `n_alts` which specifies the number of layers.

**Inputs**

  * `x`: Input that is passed to each of the layers

**Returns**

  * Output is computed by taking elementwise `max` of the outputs of the individual layers.
  * Updated state of the `layers`

**Parameters**

  * Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**States**

  * States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**References**

[1] Goodfellow, Warde-Farley, Mirza, Courville & Bengio "Maxout Networks" [https://arxiv.org/abs/1302.4389](https://arxiv.org/abs/1302.4389)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/containers.jl#L501-L545' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.NoOpLayer' href='#Lux.NoOpLayer'>#</a>&nbsp;<b><u>Lux.NoOpLayer</u></b> &mdash; <i>Type</i>.



```julia
NoOpLayer()
```

As the name suggests does nothing but allows pretty printing of layers. Whatever input is passed is returned.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/basic.jl#L83-L88' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.ReshapeLayer' href='#Lux.ReshapeLayer'>#</a>&nbsp;<b><u>Lux.ReshapeLayer</u></b> &mdash; <i>Type</i>.



```julia
ReshapeLayer(dims)
```

Reshapes the passed array to have a size of `(dims..., :)`

**Arguments**

  * `dims`: The new dimensions of the array (excluding the last dimension).

**Inputs**

  * `x`: AbstractArray of any shape which can be reshaped in `(dims..., size(x, ndims(x)))`

**Returns**

  * AbstractArray of size `(dims..., size(x, ndims(x)))`
  * Empty `NamedTuple()`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/basic.jl#L1-L18' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.SelectDim' href='#Lux.SelectDim'>#</a>&nbsp;<b><u>Lux.SelectDim</u></b> &mdash; <i>Type</i>.



```julia
SelectDim(dim, i)
```

Return a view of all the data of the input `x` where the index for dimension `dim` equals `i`. Equivalent to `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`.

**Arguments**

  * `dim`: Dimension for indexing
  * `i`: Index for dimension `dim`

**Inputs**

  * `x`: AbstractArray that can be indexed with `view(x,:,:,...,i,:,:,...)`

**Returns**

  * `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`
  * Empty `NamedTuple()`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/basic.jl#L51-L70' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.WrappedFunction' href='#Lux.WrappedFunction'>#</a>&nbsp;<b><u>Lux.WrappedFunction</u></b> &mdash; <i>Type</i>.



```julia
WrappedFunction(f)
```

Wraps a stateless and parameter less function. Might be used when a function is added to `Chain`. For example, `Chain(x -> relu.(x))` would not work and the right thing to do would be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing to do would be `Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`

**Arguments**

  * `f::Function`: A stateless and parameterless function

**Inputs**

  * `x`: s.t `hasmethod(f, (typeof(x),))` is `true`

**Returns**

  * Output of `f(x)`
  * Empty `NamedTuple()`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/basic.jl#L93-L113' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Normalization Layers'></a>

## Normalization Layers

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.BatchNorm' href='#Lux.BatchNorm'>#</a>&nbsp;<b><u>Lux.BatchNorm</u></b> &mdash; <i>Type</i>.



```julia
BatchNorm(chs::Integer, activation=identity; init_bias=zeros32, init_scale=ones32,
          affine=true, track_stats=true, epsilon=1f-5, momentum=0.1f0,
          allow_fast_activation::Bool=true)
```

[Batch Normalization](https://arxiv.org/abs/1502.03167) layer.

`BatchNorm` computes the mean and variance for each $D_1 × ... × D_{N-2} × 1 × D_N$ input slice and normalises the input accordingly.

**Arguments**

  * `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions, call the `N-1`th the channel dimension. For a batch of feature vectors this is just the data dimension, for `WHCN` images it's the usual channel dimension.
  * `activation`: After normalization, elementwise activation `activation` is applied.

**Keyword Arguments**

  * If `track_stats=true`, accumulates mean and variance statistics in training phase that will be used to renormalize the input in test phase.
  * `epsilon`: a value added to the denominator for numerical stability
  * `momentum`:  the value used for the `running_mean` and `running_var` computation
  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`
  * If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias and scale parameters.

      * `init_bias`: Controls how the `bias` is initiliazed
      * `init_scale`: Controls how the `scale` is initiliazed

**Inputs**

  * `x`: Array where `size(x, N - 1) = chs` and `ndims(x) > 2`

**Returns**

  * `y`: Normalized Array
  * Update model state

**Parameters**

  * `affine=true`

      * `bias`: Bias of shape `(chs,)`
      * `scale`: Scale of shape `(chs,)`
  * `affine=false` - Empty `NamedTuple()`

**States**

  * Statistics if `track_stats=true`

      * `running_mean`: Running mean of shape `(chs,)`
      * `running_var`: Running variance of shape `(chs,)`
  * Statistics if `track_stats=false`

      * `running_mean`: nothing
      * `running_var`: nothing
  * `training`: Used to check if training/inference mode

Use `Lux.testmode` during inference.

**Example**

```julia
m = Chain(Dense(784 => 64), BatchNorm(64, relu), Dense(64 => 10), BatchNorm(10))
```

:::warning

Passing a batch size of 1, during training will result in NaNs.

:::

See also [`BatchNorm`](layers#Lux.BatchNorm), [`InstanceNorm`](layers#Lux.InstanceNorm), [`LayerNorm`](layers#Lux.LayerNorm), [`WeightNorm`](layers#Lux.WeightNorm)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/normalize.jl#L6' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.GroupNorm' href='#Lux.GroupNorm'>#</a>&nbsp;<b><u>Lux.GroupNorm</u></b> &mdash; <i>Type</i>.



```julia
GroupNorm(chs::Integer, groups::Integer, activation=identity; init_bias=zeros32,
          init_scale=ones32, affine=true, epsilon=1f-5,
          allow_fast_activation::Bool=true)
```

[Group Normalization](https://arxiv.org/abs/1803.08494) layer.

**Arguments**

  * `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions, call the `N-1`th the channel dimension. For a batch of feature vectors this is just the data dimension, for `WHCN` images it's the usual channel dimension.
  * `groups` is the number of groups along which the statistics are computed. The number of channels must be an integer multiple of the number of groups.
  * `activation`: After normalization, elementwise activation `activation` is applied.

**Keyword Arguments**

  * `epsilon`: a value added to the denominator for numerical stability
  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`
  * If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias and scale parameters.

      * `init_bias`: Controls how the `bias` is initiliazed
      * `init_scale`: Controls how the `scale` is initiliazed

**Inputs**

  * `x`: Array where `size(x, N - 1) = chs` and `ndims(x) > 2`

**Returns**

  * `y`: Normalized Array
  * Update model state

**Parameters**

  * `affine=true`

      * `bias`: Bias of shape `(chs,)`
      * `scale`: Scale of shape `(chs,)`
  * `affine=false` - Empty `NamedTuple()`

**States**

  * `training`: Used to check if training/inference mode

Use `Lux.testmode` during inference.

**Example**

```julia
m = Chain(Dense(784 => 64), GroupNorm(64, 4, relu), Dense(64 => 10), GroupNorm(10, 5))
```

:::warning

GroupNorm doesn't have CUDNN support. The GPU fallback is not very efficient.

:::

See also [`GroupNorm`](layers#Lux.GroupNorm), [`InstanceNorm`](layers#Lux.InstanceNorm), [`LayerNorm`](layers#Lux.LayerNorm), [`WeightNorm`](layers#Lux.WeightNorm)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/normalize.jl#L146-L213' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.InstanceNorm' href='#Lux.InstanceNorm'>#</a>&nbsp;<b><u>Lux.InstanceNorm</u></b> &mdash; <i>Type</i>.



```julia
InstanceNorm(chs::Integer, activation=identity; init_bias=zeros32, init_scale=ones32,
             affine=true, epsilon=1f-5, allow_fast_activation::Bool=true)
```

Instance Normalization. For details see [1].

Instance Normalization computes the mean and variance for each $D_1 \times ... \times D_{N - 2} \times 1 \times 1$` input slice and normalises the input accordingly.

**Arguments**

  * `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions, call the `N-1`th the channel dimension. For a batch of feature vectors this is just the data dimension, for `WHCN` images it's the usual channel dimension.
  * `activation`: After normalization, elementwise activation `activation` is applied.

**Keyword Arguments**

  * `epsilon`: a value added to the denominator for numerical stability
  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`
  * If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias and scale parameters.

      * `init_bias`: Controls how the `bias` is initiliazed
      * `init_scale`: Controls how the `scale` is initiliazed

**Inputs**

  * `x`: Array where `size(x, N - 1) = chs` and `ndims(x) > 2`

**Returns**

  * `y`: Normalized Array
  * Update model state

**Parameters**

  * `affine=true`

      * `bias`: Bias of shape `(chs,)`
      * `scale`: Scale of shape `(chs,)`
  * `affine=false` - Empty `NamedTuple()`

**States**

  * `training`: Used to check if training/inference mode

Use `Lux.testmode` during inference.

**Example**

```julia
m = Chain(Dense(784 => 64), InstanceNorm(64, relu), Dense(64 => 10), InstanceNorm(10, 5))
```

**References**

[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The     missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).

:::warning

InstanceNorm doesn't have CUDNN support. The GPU fallback is not very efficient.

:::

See also [`BatchNorm`](layers#Lux.BatchNorm), [`GroupNorm`](layers#Lux.GroupNorm), [`LayerNorm`](layers#Lux.LayerNorm), [`WeightNorm`](layers#Lux.WeightNorm)


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/normalize.jl#L250' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.LayerNorm' href='#Lux.LayerNorm'>#</a>&nbsp;<b><u>Lux.LayerNorm</u></b> &mdash; <i>Type</i>.



```julia
LayerNorm(shape::NTuple{N, Int}, activation=identity; epsilon=1f-5, dims=Colon(),
          affine::Bool=true, init_bias=zeros32, init_scale=ones32,)
```

Computes mean and standard deviation over the whole input array, and uses these to normalize the whole array. Optionally applies an elementwise affine transformation afterwards.

Given an input array $x$, this layer computes

$$
y = \frac{x - \mathbb{E}[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
$$

where $\gamma$ & $\beta$ are trainable parameters if `affine=true`.

:::warning

As of v0.5.0, the doc used to say `affine::Bool=false`, but the code actually had `affine::Bool=true` as the default. Now the doc reflects the code, so please check whether your assumptions about the default (if made) were invalid.

:::

**Arguments**

  * `shape`: Broadcastable shape of input array excluding the batch dimension.
  * `activation`: After normalization, elementwise activation `activation` is applied.

**Keyword Arguments**

  * `allow_fast_activation`: If `true`, then certain activations can be approximated with a faster version. The new activation function will be given by `NNlib.fast_act(activation)`
  * `epsilon`: a value added to the denominator for numerical stability.
  * `dims`: Dimensions to normalize the array over.
  * If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias and scale parameters.

      * `init_bias`: Controls how the `bias` is initiliazed
      * `init_scale`: Controls how the `scale` is initiliazed

**Inputs**

  * `x`: AbstractArray

**Returns**

  * `y`: Normalized Array
  * Empty NamedTuple()

**Parameters**

  * `affine=false`: Empty `NamedTuple()`
  * `affine=true`

      * `bias`: Bias of shape `(shape..., 1)`
      * `scale`: Scale of shape `(shape..., 1)`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/normalize.jl#L486' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.WeightNorm' href='#Lux.WeightNorm'>#</a>&nbsp;<b><u>Lux.WeightNorm</u></b> &mdash; <i>Type</i>.



```julia
WeightNorm(layer::AbstractExplicitLayer, which_params::NTuple{N,Symbol},
           dims::Union{Tuple,Nothing}=nothing)
```

Applies [weight normalization](https://arxiv.org/abs/1602.07868) to a parameter in the given layer.

$w = g\frac{v}{\|v\|}$

Weight normalization is a reparameterization that decouples the magnitude of a weight tensor from its direction. This updates the parameters in `which_params` (e.g. `weight`) using two parameters: one specifying the magnitude (e.g. `weight_g`) and one specifying the direction (e.g. `weight_v`).

**Arguments**

  * `layer` whose parameters are being reparameterized
  * `which_params`: parameter names for the parameters being reparameterized
  * By default, a norm over the entire array is computed. Pass `dims` to modify the dimension.

**Inputs**

  * `x`: Should be of valid type for input to `layer`

**Returns**

  * Output from `layer`
  * Updated model state of `layer`

**Parameters**

  * `normalized`: Parameters of `layer` that are being normalized
  * `unnormalized`: Parameters of `layer` that are not being normalized

**States**

  * Same as that of `layer`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/normalize.jl#L361' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Upsampling'></a>

## Upsampling

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.PixelShuffle' href='#Lux.PixelShuffle'>#</a>&nbsp;<b><u>Lux.PixelShuffle</u></b> &mdash; <i>Function</i>.



```julia
PixelShuffle(r::Int)
```

Pixel shuffling layer with upscale factor `r`. Usually used for generating higher resolution images while upscaling them.

See `NNlib.pixel_shuffle` for more details.

PixelShuffle is not a Layer, rather it returns a [`WrappedFunction`](layers#Lux.WrappedFunction) with the function set to `Base.Fix2(pixel_shuffle, r)`

**Arguments**

  * `r`: Upscale factor

**Inputs**

  * `x`: For 4D-arrays representing N images, the operation converts input size(x) == (W, H, r^2 x C, N) to output of size (r x W, r x H, C, N). For D-dimensional data, it expects ndims(x) == D+2 with channel and batch dimensions, and divides the number of channels by r^D.

**Returns**

  * Output of size `(r x W, r x H, C, N)` for 4D-arrays, and `(r x W, r x H, ..., C, N)` for D-dimensional data, where `D = ndims(x) - 2`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L487-L513' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Upsample' href='#Lux.Upsample'>#</a>&nbsp;<b><u>Lux.Upsample</u></b> &mdash; <i>Type</i>.



```julia
Upsample(mode = :nearest; [scale, size]) 
Upsample(scale, mode = :nearest)
```

Upsampling Layer.

**Layer Construction**

**Option 1**

  * `mode`: Set to `:nearest`, `:linear`, `:bilinear` or `:trilinear`

Exactly one of two keywords must be specified:

  * If `scale` is a number, this applies to all but the last two dimensions (channel and batch) of the input.  It may also be a tuple, to control dimensions individually.
  * Alternatively, keyword `size` accepts a tuple, to directly specify the leading dimensions of the output.

**Option 2**

  * If `scale` is a number, this applies to all but the last two dimensions (channel and batch) of the input.  It may also be a tuple, to control dimensions individually.
  * `mode`: Set to `:nearest`, `:bilinear` or `:trilinear`

Currently supported upsampling `mode`s and corresponding NNlib's methods are:

  * `:nearest` -> `NNlib.upsample_nearest`
  * `:bilinear` -> `NNlib.upsample_bilinear`
  * `:trilinear` -> `NNlib.upsample_trilinear`

**Inputs**

  * `x`: For the input dimensions look into the documentation for the corresponding `NNlib` function

      * As a rule of thumb, `:nearest` should work with arrays of arbitrary dimensions
      * `:bilinear` works with 4D Arrays
      * `:trilinear` works with 5D Arrays

**Returns**

  * Upsampled Input of size `size` or of size `(I_1 x scale[1], ..., I_N x scale[N], C, N)`
  * Empty `NamedTuple()`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/ae33909649ecfc4c063bf731ddd25d495f6d24dc/src/layers/conv.jl#L280-L324' class='documenter-source'>source</a><br>

</div>
<br>

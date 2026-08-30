---
url: /dev/api/Lux/layers.md
---
# Built-In Layers {#Built-In-Layers}

## Containers {#Containers}

```julia
BranchLayer(layers...; fusion=nothing)
BranchLayer(; fusion=nothing, name=nothing, layers...)
```

Takes an input `x` and passes it through all the `layers` and returns a tuple of the outputs. If `fusion` is provided, applies fusion to the tuple of outputs.

**Arguments**

* Layers can be specified in two formats:
  * A list of `N` Lux layers

  * Specified as `N` keyword arguments.

**Keyword Arguments**

* `fusion`: An optional layer or function to apply to the tuple of outputs. If `fusion = nothing`, returns the tuple as-is (default behavior). If `fusion` is provided, returns `fusion((layer_1(x), layer_2(x), ..., layer_N(x)))`.

**Extended Help**

**Inputs**

* `x`: Will be directly passed to each of the `layers`

**Returns**

* If `fusion = nothing`: Tuple `(layer_1(x), layer_2(x), ..., layer_N(x))` (naming changes  if using the kwargs API)

* If `fusion` is provided: `fusion((layer_1(x), layer_2(x), ..., layer_N(x)))`

* Updated state of the `layers` (and `fusion` if it's a layer)

**Parameters**

* Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

* If `fusion` is an AbstractLuxLayer, parameters include both `layers` and `fusion`

**States**

* States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

* If `fusion` is an AbstractLuxLayer, states include both `layers` and `fusion`

::: tip Comparison with Parallel

This is slightly different from [`Parallel(nothing, layers...)`](/api/Lux/layers#Lux.Parallel)

* If the input is a tuple, `Parallel` will pass each element individually to each layer.

* `BranchLayer` essentially assumes 1 input comes in and is branched out into `N` outputs.

:::

**Example**

An easy way to replicate an input to an NTuple is to do

```julia
julia> BranchLayer(NoOpLayer(), NoOpLayer(), NoOpLayer())
BranchLayer(
    layer_(1-3) = NoOpLayer(),
)         # Total: 0 parameters,
          #        plus 0 states.
```

source

```julia
Chain(layers...; name=nothing)
Chain(; layers..., name=nothing)
```

Collects multiple layers / functions to be called in sequence on a given input.

**Arguments**

* Layers can be specified in two formats:
  * A list of `N` Lux layers

  * Specified as `N` keyword arguments.

**Extended Help**

**Inputs**

Input `x` is passed sequentially to each layer, and must conform to the input requirements of the internal layers.

**Returns**

* Output after sequentially applying all the layers to `x`

* Updated model states

**Parameters**

* Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**States**

* States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**Miscellaneous Properties**

* Allows indexing and field access syntax. We can access the `i`th layer by `m[i]` or `m.layer_i`. We can also index using ranges or arrays.

**Example**

```julia
julia> Chain(Dense(2, 3, relu), BatchNorm(3), Dense(3, 2))
Chain(
    layer_1 = Dense(2 => 3, relu),                # 9 parameters
    layer_2 = BatchNorm(3, affine=true, track_stats=true),  # 6 parameters, plus 7 non-trainable
    layer_3 = Dense(3 => 2),                      # 8 parameters
)         # Total: 23 parameters,
          #        plus 7 states.

julia> Chain(Dense(2, 3, relu), BatchNorm(3), Dense(3, 2); name="MyFancyChain")
MyFancyChain(
    layer_1 = Dense(2 => 3, relu),                # 9 parameters
    layer_2 = BatchNorm(3, affine=true, track_stats=true),  # 6 parameters, plus 7 non-trainable
    layer_3 = Dense(3 => 2),                      # 8 parameters
)         # Total: 23 parameters,
          #        plus 7 states.
```

source

```julia
PairwiseFusion(connection, layers...; name=nothing)
PairwiseFusion(connection; name=nothing, layers...)
PairwiseFusion(; connection, layers..., name=nothing)
```

```julia
x1 → layer1 → y1 ↘
                  connection → layer2 → y2 ↘
              x2 ↗                          connection → y3
                                        x3 ↗
```

**Arguments**

* `connection`: Takes 2 inputs and combines them

* `layers`: `AbstractLuxLayer`s. Layers can be specified in two formats:
  * A list of `N` Lux layers

  * Specified as `N` keyword arguments.

**Extended Help**

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

source

```julia
Parallel(connection, layers...; name=nothing)
Parallel(connection; name=nothing, layers...)
Parallel(; connection, layers..., name=nothing)
```

Create a layer which passes an input to each path in `layers`, before reducing the output with `connection`.

**Arguments**

* `connection`: An `N`-argument function that is called after passing the input through each layer, OR an AbstractLuxLayer that takes a tuple of `N` inputs.  If `connection = nothing`, we return a tuple: `Parallel(nothing, f, g)(x, y) = (f(x), g(y))`

* Layers can be specified in two formats:
  * A list of `N` Lux layers

  * Specified as `N` keyword arguments.

**Extended Help**

**Inputs**

* `x`: If `x` is not a tuple, then return is computed as `connection([l(x) for l in layers]...)`. Else one is passed to each layer, thus `Parallel(+, f, g)(x, y) = f(x) + g(y)`.

**Returns**

* See the Inputs section for how the output is computed

* Updated state of the `layers` (and `connection` if it's a layer)

**Parameters**

* Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

* If `connection` is an AbstractLuxLayer, parameters include both `layers` and `connection`

**States**

* States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

* If `connection` is an AbstractLuxLayer, states include both `layers` and `connection`

See also [`SkipConnection`](/api/Lux/layers#Lux.SkipConnection) which is `Parallel` with one identity.

**Example**

```julia
julia> model = Parallel(nothing, Dense(2, 1), Dense(2, 1))
Parallel(
    layer_(1-2) = Dense(2 => 1),                  # 6 (3 x 2) parameters
)         # Total: 6 parameters,
          #        plus 0 states.

julia> using Random;
       rng = Random.seed!(123);
       ps, st = Lux.setup(rng, model);
       x1 = randn(rng, Float32, 2);
       x2 = randn(rng, Float32, 2);

julia> size.(first(model((x1, x2), ps, st)))
((1,), (1,))
```

source

```julia
SkipConnection(layers, connection; name=nothing)
SkipConnection(; layers, connection, name=nothing)
```

Create a skip connection which consists of a layer or [`Chain`](/api/Lux/layers#Lux.Chain) of consecutive layers and a shortcut connection linking the block's input to the output through a user-supplied 2-argument callable. The first argument to the callable will be propagated through the given `layer` while the second is the unchanged, "skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`.

**Arguments**

* `layer`: Layer or `Chain` of layers to be applied to the input

* `connection`:
  * A 2-argument function that takes `layer(input)` and the input OR

  * An AbstractLuxLayer that takes `(layer(input), input)` as input

**Extended Help**

**Inputs**

* `x`: Will be passed directly to `layer`

**Returns**

* Output of `connection(layer(input), input)`

* Updated state of `layer`

**Parameters**

* Parameters of `layer` OR

* If `connection` is an AbstractLuxLayer, then NamedTuple with fields `:layers` and `:connection`

**States**

* States of `layer` OR

* If `connection` is an AbstractLuxLayer, then NamedTuple with fields `:layers` and `:connection`

See [`Parallel`](/api/Lux/layers#Lux.Parallel) for a more general implementation.

source

```julia
RepeatedLayer(model; repeats::Val = Val(10), input_injection::Val = Val(false))
```

Iteratively applies `model` for `repeats` number of times. The initial input is passed into the model repeatedly if `input_injection = Val(true)`. This layer unrolls the computation, however, semantically this is same as:

* `input_injection = Val(false)`

  ```julia
  res = x
  for i in 1:repeats
      res, st = model(res, ps, st)
  end
  ```

* `input_injection = Val(true)`

  ```julia
  res = x
  for i in 1:repeats
      res, st = model((res, x), ps, st)
  end
  ```

It is expected that `repeats` will be a reasonable number below `20`, beyond that compile times for gradients might be unreasonably high.

**Arguments**

* `model` must be an `AbstractLuxLayer`

**Keyword Arguments**

* `repeats`: Number of times to apply the model

* `input_injection`: If `true`, then the input is passed to the model along with the output

**Extended Help**

**Inputs**

* `x`: Input as described above

**Returns**

* Output is computed by as described above

* Updated state of the `model`

**Parameters**

* Parameters of `model`

**States**

* State of `model`

source

```julia
AlternatePrecision{T}(layer)
AlternatePrecision(::Type{T}, layer)
```

This layer is used to convert the input to a different precision (`T`), execute the layer, and then convert the output back to the original precision.

**Arguments**

* `T`: The eltype of the input to the layer

* `layer`: The layer to execute

**Inputs**

* `x`: AbstractArray

**Returns**

* `y`: Output of the layer

* State of the output

source

## Convolutional Layers {#Convolutional-Layers}

```julia
Conv(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
     activation=identity; init_weight=nothing, init_bias=nothing, stride=1,
     pad=0, dilation=1, groups=1, use_bias=True(), cross_correlation=False())
```

Standard convolutional layer.

::: tip Conv 2D

Image data should be stored in WHCN order (width, height, channels, batch). In other words, a `100 x 100` RGB image would be a `100 x 100 x 3 x 1` array, and a batch of 50 would be a `100 x 100 x 3 x 50` array. This has `N = 2` spatial dimensions, and needs a kernel size like `(5, 5)`, a 2-tuple of integers. To take convolutions along `N` feature dimensions, this layer expects as input an array with `ndims(x) == N + 2`, where `size(x, N + 1) == in_chs` is the number of input channels, and `size(x, ndims(x))` is the number of observations in a batch.

:::

::: warning Warning

Frameworks like [`Pytorch`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d) perform cross-correlation in their convolution layers. Pass `cross_correlation=true` to use cross-correlation instead.

:::

**Arguments**

* `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D      convolutions `length(k) == 2`

* `in_chs`: Number of input channels

* `out_chs`: Number of input and output channels

* `activation`: Activation Function

**Extended Help**

**Keyword Arguments**

* `init_weight`: Controls the initialization of the weight parameter. If `nothing`, then we use [`kaiming_uniform`](/api/Building_Blocks/WeightInitializers#WeightInitializers.kaiming_uniform) with gain computed on the basis of the activation function (taken from Pytorch [`nn.init.calculate_gain`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain)).

* `init_bias`: Controls the initialization of the bias parameter. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(fan_in))`.

* `stride`: Should each be either single integer, or a tuple with `N` integers

* `dilation`: Should each be either single integer, or a tuple with `N` integers

* `pad`: Specifies the number of elements added to the borders of the data array. It can        be
  * a single integer for equal padding all around,

  * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,

  * a tuple of `2*N` integers, for asymmetric padding, or

  * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.

  * Periodic padding can achieved by pre-empting the layer with a `WrappedFunction(x -> NNlib.pad_circular(x, N_pad; dims=pad_dims))`

* `groups`: Expected to be an `Int`. It specifies the number of groups to divide a           convolution into (set `groups = in_chs` for Depthwise Convolutions). `in_chs`           and `out_chs` must be divisible by `groups`.

* `use_bias`: Trainable bias can be disabled entirely by setting this to `false`.

* `cross_correlation`: If `true`, perform cross-correlation instead of convolution. Prior to `v1`, Lux used to have a `CrossCor` layer which performed cross-correlation. This was removed in `v1` in favor of `Conv` with `cross_correlation=true`.

**Inputs**

* `x`: Data satisfying `ndims(x) == N + 2 && size(x, N - 1) == in_chs`, i.e.      `size(x) = (I_N, ..., I_1, C_in, N)`

**Returns**

* Output of the convolution `y` of size `(O_N, ..., O_1, C_out, N)` where

$$O\_i = \left\lfloor\frac{I\_i + p\_i + p\_{(i + N) % |p|} - d\_i \times (k\_i - 1)}{s\_i} + 1\right\rfloor$$

* Empty `NamedTuple()`

**Parameters**

* `weight`: Convolution kernel

* `bias`: Bias (present if `use_bias=true`)

source

```julia
ConvTranspose(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
              activation=identity; init_weight=glorot_uniform, init_bias=zeros32,
              stride=1, pad=0, outpad=0, dilation=1, groups=1, use_bias=True(),
              cross_correlation=False())
```

Standard convolutional transpose layer.

**Arguments**

* `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D      convolutions `length(k) == 2`

* `in_chs`: Number of input channels

* `out_chs`: Number of input and output channels

* `activation`: Activation Function

**Keyword Arguments**

* `init_weight`: Controls the initialization of the weight parameter. If `nothing`, then we use [`kaiming_uniform`](/api/Building_Blocks/WeightInitializers#WeightInitializers.kaiming_uniform) with gain computed on the basis of the activation function (taken from Pytorch [`nn.init.calculate_gain`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain)).

* `init_bias`: Controls the initialization of the bias parameter. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(fan_in))`.

* `stride`: Should each be either single integer, or a tuple with `N` integers

* `dilation`: Should each be either single integer, or a tuple with `N` integers

* `pad`: Specifies the number of elements added to the borders of the data array. It can        be
  * a single integer for equal padding all around,

  * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,

  * a tuple of `2*N` integers, for asymmetric padding, or

  * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) * stride` (possibly rounded) for each spatial dimension.

* `groups`: Expected to be an `Int`. It specifies the number of groups to divide a           convolution into (set `groups = in_chs` for Depthwise Convolutions). `in_chs`           and `out_chs` must be divisible by `groups`.

* `use_bias`: Trainable bias can be disabled entirely by setting this to `false`.

* `cross_correlation`: If `true`, perform transposed cross-correlation instead of transposed convolution.

* `outpad`: To converse [`Conv`](/api/Lux/layers#Lux.Conv) inversability when `stride > 1`, `outpad` can be used to increase the size of the output in the desired dimensions. Whereas `pad` is used to zero-pad the input, `outpad` only affects the output shape.

**Extended Help**

**Inputs**

* `x`: Data satisfying `ndims(x) == N + 2 && size(x, N - 1) == in_chs`, i.e.      `size(x) = (I_N, ..., I_1, C_in, N)`

**Returns**

* Output of the convolution transpose `y` of size `(O_N, ..., O_1, C_out, N)` where

* Empty `NamedTuple()`

**Parameters**

* `weight`: Convolution Transpose kernel

* `bias`: Bias (present if `use_bias=true`)

source

## Dropout Layers {#Dropout-Layers}

```julia
AlphaDropout(p::Real)
```

AlphaDropout layer.

**Arguments**

* `p`: Probability of Dropout
  * if `p = 0` then [`NoOpLayer`](/api/Lux/layers#Lux.NoOpLayer) is returned.

  * if `p = 1` then `WrappedLayer(Base.Fix1(broadcast, zero))` is returned.

**Inputs**

* `x`: Must be an AbstractArray

**Returns**

* `x` with dropout mask applied if `training=Val(true)` else just `x`

* State with updated `rng`

**States**

* `rng`: Pseudo Random Number Generator

* `training`: Used to check if training/inference mode

Call [`Lux.testmode`](/api/Building_Blocks/LuxCore#LuxCore.testmode) to switch to test mode.

See also [`Lux.Dropout`](/api/Lux/layers#Lux.Dropout), [`VariationalHiddenDropout`](/api/Lux/layers#Lux.VariationalHiddenDropout)

source

```julia
Dropout(p; dims=:)
```

Dropout layer.

**Arguments**

* `p`: Probability of Dropout (if `p = 0` then [`NoOpLayer`](/api/Lux/layers#Lux.NoOpLayer) is returned)

**Keyword Arguments**

* To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `Dropout(p; dims = (3,4))` will randomly zero out entire channels on WHCN input (also called 2D dropout).

**Inputs**

* `x`: Must be an AbstractArray

**Returns**

* `x` with dropout mask applied if `training=Val(true)` else just `x`

* State with updated `rng`

**States**

* `rng`: Pseudo Random Number Generator

* `training`: Used to check if training/inference mode

Call [`Lux.testmode`](/api/Building_Blocks/LuxCore#LuxCore.testmode) to switch to test mode.

See also [`AlphaDropout`](/api/Lux/layers#Lux.AlphaDropout), [`VariationalHiddenDropout`](/api/Lux/layers#Lux.VariationalHiddenDropout)

source

```julia
VariationalHiddenDropout(p; dims=:)
```

VariationalHiddenDropout layer. The only difference from Dropout is that the `mask` is retained until [`Lux.update_state(l, :update_mask, Val(true))`](/api/Building_Blocks/LuxCore#LuxCore.update_state) is called.

**Arguments**

* `p`: Probability of Dropout (if `p = 0` then [`NoOpLayer`](/api/Lux/layers#Lux.NoOpLayer) is returned)

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

Call [`Lux.testmode`](/api/Building_Blocks/LuxCore#LuxCore.testmode) to switch to test mode.

See also [`AlphaDropout`](/api/Lux/layers#Lux.AlphaDropout), [`Lux.Dropout`](/api/Lux/layers#Lux.Dropout)

source

## Pooling Layers {#Pooling-Layers}

```julia
AdaptiveLPPool(output_size; p=2)
```

Adaptive LP Pooling layer. Calculates the necessary window size such that its output has `size(y)[1:N] == output_size`.

**Arguments**

* `output_size`: Size of the first `N` dimensions for the output

::: danger GPU Support

This layer is currently only supported on CPU.

:::

**Inputs**

* `x`: Expects as input an array with `ndims(x) == N + 2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(output_size)`.

**Returns**

* Output of size `(out..., C, N)`

* Empty `NamedTuple()`

source

```julia
AdaptiveMaxPool(output_size)
```

Adaptive Max Pooling layer. Calculates the necessary window size such that its output has `size(y)[1:N] == output_size`.

**Arguments**

* `output_size`: Size of the first `N` dimensions for the output

**Inputs**

* `x`: Expects as input an array with `ndims(x) == N + 2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(output_size)`.

**Returns**

* Output of size `(out..., C, N)`

* Empty `NamedTuple()`

source

```julia
AdaptiveMeanPool(output_size)
```

Adaptive Mean Pooling layer. Calculates the necessary window size such that its output has `size(y)[1:N] == output_size`.

**Arguments**

* `output_size`: Size of the first `N` dimensions for the output

**Inputs**

* `x`: Expects as input an array with `ndims(x) == N + 2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(output_size)`.

**Returns**

* Output of size `(out..., C, N)`

* Empty `NamedTuple()`

source

```julia
GlobalLPPool(; p=2)
```

Global LP Pooling layer. Transforms `(w, h, c, b)`-shaped input into `(1, 1, c, b)`-shaped output, by performing mean pooling on the complete `(w, h)`-shaped feature maps.

::: danger GPU Support

This layer is currently only supported on CPU.

:::

**Inputs**

* `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

* Output of the pooling `y` of size `(1, ..., 1, C, N)`

* Empty `NamedTuple()`

source

```julia
GlobalMaxPool()
```

Global Max Pooling layer. Transforms `(w, h, c, b)`-shaped input into `(1, 1, c, b)`-shaped output, by performing mean pooling on the complete `(w, h)`-shaped feature maps.

**Inputs**

* `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

* Output of the pooling `y` of size `(1, ..., 1, C, N)`

* Empty `NamedTuple()`

source

```julia
GlobalMeanPool()
```

Global Mean Pooling layer. Transforms `(w, h, c, b)`-shaped input into `(1, 1, c, b)`-shaped output, by performing mean pooling on the complete `(w, h)`-shaped feature maps.

**Inputs**

* `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

* Output of the pooling `y` of size `(1, ..., 1, C, N)`

* Empty `NamedTuple()`

source

```julia
LPPool(window; stride=window, pad=0, dilation=1, p=2)
```

LP Pooling layer, which replaces all pixels in a block of size `window` with the reduction operation: lp.

**Arguments**

* `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling `length(window) == 2`

**Keyword Arguments**

* `stride`: Should each be either single integer, or a tuple with `N` integers

* `dilation`: Should each be either single integer, or a tuple with `N` integers

* `pad`: Specifies the number of elements added to the borders of the data array. It can be
  * a single integer for equal padding all around,

  * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,

  * a tuple of `2*N` integers, for asymmetric padding, or

  * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.

::: danger GPU Support

This layer is currently only supported on CPU.

:::

**Extended Help**

**Inputs**

* `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

* Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where

$$    O\_i = \left\lfloor\frac{I\_i + p\_i + p\_{(i + N) % |p|} - d\_i \times (k\_i - 1)}{s\_i} + 1\right\rfloor$$

* Empty `NamedTuple()`

source

```julia
MaxPool(window; stride=window, pad=0, dilation=1)
```

Max Pooling layer, which replaces all pixels in a block of size `window` with the reduction operation: max.

**Arguments**

* `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling `length(window) == 2`

**Keyword Arguments**

* `stride`: Should each be either single integer, or a tuple with `N` integers

* `dilation`: Should each be either single integer, or a tuple with `N` integers

* `pad`: Specifies the number of elements added to the borders of the data array. It can be
  * a single integer for equal padding all around,

  * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,

  * a tuple of `2*N` integers, for asymmetric padding, or

  * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.

**Extended Help**

**Inputs**

* `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

* Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where

$$    O\_i = \left\lfloor\frac{I\_i + p\_i + p\_{(i + N) % |p|} - d\_i \times (k\_i - 1)}{s\_i} + 1\right\rfloor$$

* Empty `NamedTuple()`

source

```julia
MeanPool(window; stride=window, pad=0, dilation=1)
```

Mean Pooling layer, which replaces all pixels in a block of size `window` with the reduction operation: mean.

**Arguments**

* `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling `length(window) == 2`

**Keyword Arguments**

* `stride`: Should each be either single integer, or a tuple with `N` integers

* `dilation`: Should each be either single integer, or a tuple with `N` integers

* `pad`: Specifies the number of elements added to the borders of the data array. It can be
  * a single integer for equal padding all around,

  * a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,

  * a tuple of `2*N` integers, for asymmetric padding, or

  * the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.

**Extended Help**

**Inputs**

* `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

**Returns**

* Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where

$$    O\_i = \left\lfloor\frac{I\_i + p\_i + p\_{(i + N) % |p|} - d\_i \times (k\_i - 1)}{s\_i} + 1\right\rfloor$$

* Empty `NamedTuple()`

source

## Recurrent Layers {#Recurrent-Layers}

```julia
GRUCell((in_dims, out_dims)::Pair{<:Int,<:Int}; use_bias=true, train_state::Bool=false,
        init_weight=glorot_uniform, init_recurrent_weight=init_weight,
        init_bias=nothing, init_state=zeros32)
```

Gated Recurrent Unit (GRU) Cell

$$\begin{align}
r &= \sigma(W\_{ir} \times x + b\_{ir} + W\_{hr} \times h\_{prev} + b\_{hr})\\
z &= \sigma(W\_{iz} \times x + b\_{iz} + W\_{hz} \times h\_{prev} + b\_{hz})\\
n &= \tanh(W\_{in} \times x + b\_{in} + r \cdot (W\_{hn} \times h\_{prev} + b\_{hn}))\\
h\_{new} &= (1 - z) \cdot n + z \cdot h\_{prev}
\end{align}$$

**Arguments**

* `in_dims`: Input Dimension

* `out_dims`: Output (Hidden State) Dimension

* `use_bias`: Set to false to deactivate bias

* `train_state`: Trainable initial hidden state can be activated by setting this to `true`

* `init_bias`: Initializer for bias. Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3 element tuple. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(out_dims))`.

* `init_weight`: Initializer for weight. Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3 element tuple. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(out_dims))`.

* `init_recurrent_weight`: Initializer for weight. Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3 element tuple. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(out_dims))`.

* `init_state`: Initializer for hidden state

**Inputs**

* Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `false` - Creates a hidden state using `init_state` and proceeds to Case 2.

* Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `true` - Repeats `hidden_state` from parameters to match the shape of `x`          and proceeds to Case 2.

* Case 2: Tuple `(x, (h, ))` is provided, then the output and a tuple containing the          updated hidden state is returned.

**Returns**

* Tuple containing
  * Output $h\_{new}$ of shape `(out_dims, batch_size)`

  * Tuple containing new hidden state $h\_{new}$

* Updated model state

**Parameters**

* `weight_ih`: Concatenated Weights to map from input space              ${ W\_{ir}, W\_{iz}, W\_{in} }$.

* `weight_hh`: Concatenated Weights to map from hidden space              ${ W\_{hr}, W\_{hz}, W\_{hn} }$.

* `bias_ih`: Concatenated Bias vector for the input space            ${ b\_{ir}, b\_{iz}, b\_{in} }$ (not present if `use_bias=false`).

* `bias_hh`: Concatenated Bias vector for the hidden space            ${ b\_{hr}, b\_{hz}, b\_{hn} }$ (not present if `use_bias=false`).

* `hidden_state`: Initial hidden state vector (not present if `train_state=false`)           ${ b\_{hr}, b\_{hz}, b\_{hn} }$.

**States**

* `rng`: Controls the randomness (if any) in the initial state generation

source

```julia
LSTMCell(in_dims => out_dims; use_bias::Bool=true, train_state::Bool=false,
         train_memory::Bool=false, init_weight=nothing,
         init_recurrent_weight=init_weight,
         init_bias=nothing, init_state=zeros32, init_memory=zeros32)
```

Long Short-Term (LSTM) Cell

$$\begin{align}
i &= \sigma(W\_{ii} \times x + W\_{hi} \times h\_{prev} + b\_{i})\\
f &= \sigma(W\_{if} \times x + W\_{hf} \times h\_{prev} + b\_{f})\\
g &= \tanh(W\_{ig} \times x + W\_{hg} \times h\_{prev} + b\_{g})\\
o &= \sigma(W\_{io} \times x + W\_{ho} \times h\_{prev} + b\_{o})\\
c\_{new} &= f \cdot c\_{prev} + i \cdot g\\
h\_{new} &= o \cdot \tanh(c\_{new})
\end{align}$$

**Arguments**

* `in_dims`: Input Dimension

* `out_dims`: Output (Hidden State & Memory) Dimension

* `use_bias`: Set to false to deactivate bias

* `train_state`: Trainable initial hidden state can be activated by setting this to `true`

* `train_memory`: Trainable initial memory can be activated by setting this to `true`

* `init_bias`: Initializer for bias. Must be a tuple containing 4 functions. If a single value is passed, it is copied into a 4 element tuple. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(out_dims))`.

* `init_weight`: Initializer for weight. Must be a tuple containing 4 functions. If a single value is passed, it is copied into a 4 element tuple. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(out_dims))`.

* `init_recurrent_weight`: Initializer for recurrent weight. Must be a tuple containing 4 functions. If a single value is passed, it is copied into a 4 element tuple. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(out_dims))`.

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
  * Output $h\_{new}$ of shape `(out_dims, batch_size)`

  * Tuple containing new hidden state $h\_{new}$ and new memory $c\_{new}$

* Updated model state

**Parameters**

* `weight_ih`: Concatenated Weights to map from input space              ${ W\_{ii}, W\_{if}, W\_{ig}, W\_{io} }$.

* `weight_hh`: Concatenated Weights to map from hidden space              ${ W\_{hi}, W\_{hf}, W\_{hg}, W\_{ho} }$

* `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)

* `bias_hh`: Concatenated Bias vector for the hidden-hidden connection (not present if `use_bias=false`)

* `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

* `memory`: Initial memory vector (not present if `train_memory=false`)

**States**

* `rng`: Controls the randomness (if any) in the initial state generation

source

```julia
RNNCell(in_dims => out_dims, activation=tanh; use_bias=True(), train_state=False(),
    init_bias=nothing, init_weight=nothing, init_recurrent_weight=init_weight,
    init_state=zeros32)
```

An Elman RNNCell cell with `activation` (typically set to `tanh` or `relu`).

$h\_{new} = activation(weight\_{ih} \times x + bias\_{ih} + weight\_{hh} \times h\_{prev} + bias\_{hh})$

**Arguments**

* `in_dims`: Input Dimension

* `out_dims`: Output (Hidden State) Dimension

* `activation`: Activation function

* `use_bias`: Set to false to deactivate bias

* `train_state`: Trainable initial hidden state can be activated by setting this to `true`

* `init_bias`: Initializer for bias. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(out_dims))`.

* `init_weight`: Initializer for weight. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(out_dims))`.

* `init_recurrent_weight`: Initializer for recurrent weight. If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(out_dims))`.

* `init_state`: Initializer for hidden state

**Inputs**

* Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `false` - Creates a hidden state using `init_state` and proceeds to Case 2.

* Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set          to `true` - Repeats `hidden_state` from parameters to match the shape of `x`          and proceeds to Case 2.

* Case 2: Tuple `(x, (h, ))` is provided, then the output and a tuple containing the         updated hidden state is returned.

**Returns**

* Tuple containing
  * Output $h\_{new}$ of shape `(out_dims, batch_size)`

  * Tuple containing new hidden state $h\_{new}$

* Updated model state

**Parameters**

* `weight_ih`: Maps the input to the hidden state.

* `weight_hh`: Maps the hidden state to the hidden state.

* `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)

* `bias_hh`: Bias vector for the hidden-hidden connection (not present if `use_bias=false`)

* `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

**States**

* `rng`: Controls the randomness (if any) in the initial state generation

source

```julia
Recurrence(
    cell;
    ordering::AbstractTimeSeriesDataBatchOrdering=BatchLastIndex(),
    return_sequence::Bool=false,
    mincut::Bool=false,
)
```

Wraps a recurrent cell (like [`RNNCell`](/api/Lux/layers#Lux.RNNCell), [`LSTMCell`](/api/Lux/layers#Lux.LSTMCell), [`GRUCell`](/api/Lux/layers#Lux.GRUCell)) to automatically operate over a sequence of inputs.

::: warning Relation to `Flux.Recur`

This is completely distinct from `Flux.Recur`. It doesn't make the `cell` stateful, rather allows operating on an entire sequence of inputs at once. See [`StatefulRecurrentCell`](/api/Lux/layers#Lux.StatefulRecurrentCell) for functionality similar to `Flux.Recur`.

:::

**Arguments**

* `cell`: A recurrent cell. See [`RNNCell`](/api/Lux/layers#Lux.RNNCell), [`LSTMCell`](/api/Lux/layers#Lux.LSTMCell), [`GRUCell`](/api/Lux/layers#Lux.GRUCell), for how the inputs/outputs of a recurrent cell must be structured.

**Keyword Arguments**

* `return_sequence`: If `true` returns the entire sequence of outputs, else returns only the last output. Defaults to `false`.

* `ordering`: The ordering of the batch and time dimensions in the input. Defaults to `BatchLastIndex()`. Alternatively can be set to `TimeLastIndex()`.

* `mincut`: If `true`, we will using mincut for the reverse mode differentiation. *(Only for Reactant)*

**Extended Help**

**Inputs**

* If `x` is a
  * Tuple or Vector: Each element is fed to the `cell` sequentially.

  * Array (except a Vector): It is spliced along the penultimate dimension and each slice is fed to the `cell` sequentially.

**Returns**

* Output of the `cell` for the entire sequence.

* Update state of the `cell`.

::: tip Tip

Frameworks like Tensorflow have special implementation of [`StackedRNNCells`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/StackedRNNCells) to handle sequentially composed RNN Cells. In Lux, one can simple stack multiple `Recurrence` blocks in a `Chain` to achieve the same.

```julia
Chain(
    Recurrence(RNNCell(inputsize => latentsize); return_sequence=true),
    Recurrence(RNNCell(latentsize => latentsize); return_sequence=true),
    :
    x -> stack(x; dims=2)
)
```

For some discussion on this topic, see https://github.com/LuxDL/Lux.jl/issues/472.

:::

source

```julia
StatefulRecurrentCell(cell)
```

Wraps a recurrent cell (like [`RNNCell`](/api/Lux/layers#Lux.RNNCell), [`LSTMCell`](/api/Lux/layers#Lux.LSTMCell), [`GRUCell`](/api/Lux/layers#Lux.GRUCell)) and makes it stateful.

To avoid undefined behavior, once the processing of a single sequence of data is complete, update the state with `Lux.update_state(st, :carry, nothing)`.

**Arguments**

* `cell`: A recurrent cell. See [`RNNCell`](/api/Lux/layers#Lux.RNNCell), [`LSTMCell`](/api/Lux/layers#Lux.LSTMCell), [`GRUCell`](/api/Lux/layers#Lux.GRUCell), for how the inputs/outputs of a recurrent cell must be structured.

**Inputs**

* Input to the `cell`.

**Returns**

* Output of the `cell` for the entire sequence.

* Update state of the `cell` and updated `carry`.

**States**

* NamedTuple containing:
  * `cell`: Same as `cell`.

  * `carry`: The carry state of the `cell`.

source

```julia
BidirectionalRNN(cell::AbstractRecurrentCell,
    backward_cell::Union{AbstractRecurrentCell, Nothing}=nothing;
    merge_mode::Union{Function, Nothing}=vcat,
    ordering::AbstractTimeSeriesDataBatchOrdering=BatchLastIndex())
```

Bidirectional RNN wrapper.

**Arguments**

* `cell`: A recurrent cell. See [`RNNCell`](/api/Lux/layers#Lux.RNNCell), [`LSTMCell`](/api/Lux/layers#Lux.LSTMCell), [`GRUCell`](/api/Lux/layers#Lux.GRUCell), for how the inputs/outputs of a recurrent cell must be structured.

* `backward_cell`: A optional backward recurrent cell. If `backward_cell` is `nothing`, the rnn layer instance passed as the `cell` argument will be used to generate the backward layer automatically. `in_dims` of `backward_cell` should be consistent with `in_dims` of `cell`

**Keyword Arguments**

* `merge_mode`: Function by which outputs of the forward and backward RNNs will be combined. default value is `vcat`. If `nothing`, the outputs will not be combined.

* `ordering`: The ordering of the batch and time dimensions in the input. Defaults to `BatchLastIndex()`. Alternatively can be set to `TimeLastIndex()`.

**Extended Help**

**Inputs**

* If `x` is a
  * Tuple or Vector: Each element is fed to the `cell` sequentially.

  * Array (except a Vector): It is spliced along the penultimate dimension and each slice is fed to the `cell` sequentially.

**Returns**

* Merged output of the `cell` and `backward_cell` for the entire sequence.

* Update state of the `cell` and `backward_cell`.

**Parameters**

* `NamedTuple` with `cell` and `backward_cell`.

**States**

* Same as `cell` and `backward_cell`.

source

## Linear Layers {#Linear-Layers}

```julia
Bilinear((in1_dims, in2_dims) => out, activation=identity; init_weight=nothing,
         init_bias=nothing, use_bias=True())
Bilinear(in12_dims => out, activation=identity; init_weight=nothing,
         init_bias=nothing, use_bias=True())
```

Create a fully connected layer between two inputs and an output, and otherwise similar to [`Dense`](/api/Lux/layers#Lux.Dense). Its output, given vectors `x` & `y`, is another vector `z` with, for all `i in 1:out`:

`z[i] = activation(x' * W[i, :, :] * y + bias[i])`

If `x` and `y` are matrices, then each column of the output `z = B(x, y)` is of this form, with `B` the Bilinear layer.

**Arguments**

* `in1_dims`: number of input dimensions of `x`

* `in2_dims`: number of input dimensions of `y`

* `in12_dims`: If specified, then `in1_dims = in2_dims = in12_dims`

* `out`: number of output dimensions

* `activation`: activation function

**Keyword Arguments**

* `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in1_dims, in2_dims)`). If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(in1_dims))`.

* `init_bias`: initializer for the bias vector (ignored if `use_bias=false`). If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(in1_dims))`.

* `use_bias`: Trainable bias can be disabled entirely by setting this to `false`

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

source

```julia
Dense(in_dims => out_dims, activation=identity; init_weight=nothing,
      init_bias=nothing, use_bias=True())
```

Create a traditional fully connected layer, whose forward pass is given by: `y = activation.(weight * x .+ bias)`

**Arguments**

* `in_dims`: number of input dimensions

* `out_dims`: number of output dimensions

* `activation`: activation function

**Keyword Arguments**

* `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in_dims)`). If `nothing`, then we use [`kaiming_uniform`](/api/Building_Blocks/WeightInitializers#WeightInitializers.kaiming_uniform) with gain computed on the basis of the activation function (taken from Pytorch [`nn.init.calculate_gain`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain)).

* `init_bias`: initializer for the bias vector (ignored if `use_bias=false`). If `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where `bound = inv(sqrt(in_dims))`.

* `use_bias`: Trainable bias can be disabled entirely by setting this to `false`

**Input**

* `x` must be an AbstractArray with `size(x, 1) == in_dims`

**Returns**

* AbstractArray with dimensions `(out_dims, ...)` where `...` are the dimensions of `x`

* Empty `NamedTuple()`

**Parameters**

* `weight`: Weight Matrix of size `(out_dims, in_dims)`

* `bias`: Bias of size `(out_dims, 1)` (present if `use_bias=true`)

source

```julia
Scale(dims, activation=identity; init_weight=ones32, init_bias=zeros32, use_bias=True())
```

Create a Sparsely Connected Layer with a very specific structure (only Diagonal Elements are non-zero). The forward pass is given by: `y = activation.(weight .* x .+ bias)`

**Arguments**

* `dims`: size of the learnable scale and bias parameters.

* `activation`: activation function

**Keyword Arguments**

* `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in_dims)`)

* `init_bias`: initializer for the bias vector (ignored if `use_bias=false`)

* `use_bias`: Trainable bias can be disabled entirely by setting this to `false`

**Input**

* `x` must be an Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])` for `k ≤ size(dims)`

**Returns**

* Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])` for `k ≤ size(dims)`

* Empty `NamedTuple()`

**Parameters**

* `weight`: Weight Array of size `(dims...)`

* `bias`: Bias of size `(dims...)`

source

## Attention Layers {#Attention-Layers}

```julia
MultiHeadAttention(dims; nheads=1, dense_kwargs=(; use_bias=False()),
                   attention_dropout_probability=0.0f0,
                   is_causal::Union{Bool,Nothing}=nothing)
```

The multi-head dot-product attention layer used in Transformer architectures \[[1](/references#vaswani2017attention)].

**Arguments**

* `dims`: The embedding dimensions of inputs, intermediate tensors and outputs. In the most general case, it is given as

  * a) `(q_in_dim, k_in_dim, v_in_dim) => (qk_dim, v_dim) => out_dim`.

  Can take also simpler forms as

  * b) `dims::Int`;

  * c) `in_dim::Int => (qk_dim, v_dim) => out_dim`;

  * d) `in_dim::Int => qkv_dim => out_dim`.

**Keyword Arguments**

* `nheads`: number of heads.

* `attention_dropout_probability`: dropout probability for the attention scores.

* `dense_kwargs`: keyword arguments for the Dense layers. Default `use_bias=false`.

* `is_causal`: whether the attention is causal. If this is provided, the attention mask will be automatically created (passing in the `mask` argument is not allowed and will throw an error).

**Forward Pass Signature(s)**

```julia
(m::MultiHeadAttention)(qkv, ps, st::NamedTuple)
(m::MultiHeadAttention)((q, kv), ps, st::NamedTuple)
(m::MultiHeadAttention)((q, k, v, [mask = nothing]), ps, st::NamedTuple)
```

**Inputs**

* `qkv`: a single input tensor for query, key and value. This corresponds to self-attention.

* `(q, kv)`: a tuple of two input tensors for query and key-value.

* `(q, k, v)`: a tuple of three input tensors for query, key and value.

* `mask`: an optional mask to apply to the attention scores. This must be broadcastable to the shape of the attention scores `(kv_len, q_len, nheads, batch_size)`.

The query tensor `q` is expected to have shape `(q_in_dim, q_len, batch_size)`, the key test `k` is expected to have shape `(k_in_dim, kv_len, batch_size)`, the value tensor `v` is expected to have shape `(v_in_dim, kv_len, batch_size)`.

**Returns**

* A tuple of two elements. The first element is the output tensor of shape `(out_dim, q_len, batch_size)` and the second element is the attention scores of shape `(q_len, kv_len, nheads, batch_size)`.

* A NamedTuple of the states of the layer.

**Extended Help**

**Examples**

```julia
julia> m = MultiHeadAttention(64; nheads=8);

julia> ps, st = Lux.setup(Random.default_rng(), m);

julia> q = randn(Float32, 64, 10, 32);

julia> k = randn(Float32, 64, 20, 32);

julia> v = randn(Float32, 64, 20, 32);

julia> (y, α), st_new = m((q, k, v), ps, st);

julia> size(y)
(64, 10, 32)

julia> size(α)
(20, 10, 8, 32)

julia> (y, α), st_new = m(q, ps, st);  # self-attention

julia> size(y)
(64, 10, 32)

julia> size(α)
(10, 10, 8, 32)
```

source

## Embedding Layers {#Embedding-Layers}

```julia
Embedding(in_dims => out_dims; init_weight=rand32)
```

A lookup table that stores embeddings of dimension `out_dims` for a vocabulary of size `in_dims`. When the vocabulary is multi-dimensional, the input is expected to be a tuple of Cartesian indices.

This layer is often used to store word embeddings and retrieve them using indices.

**Arguments**

* `in_dims`: number(s) of input dimensions

* `out_dims`: number of output dimensions

**Keyword Arguments**

* `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in_dims...)`)

**Input**

* Integer OR

* Abstract Vector of Integers OR

* Abstract Array of Integers OR

* Tuple of Integers OR

* Tuple of Abstract Vectors of Integers OR

* Tuple of Abstract Arrays of Integers

**Returns**

* Returns the embedding corresponding to each index in the input. For an N dimensional input, an N + 1 dimensional output is returned.

* Empty `NamedTuple()`

::: warning Gradients with Tracker.jl

Tracker.jl produces incorrect gradients for this layer if indices in the input are repeated. Don't use this layer with Tracker.jl if you need to compute gradients.

:::

source

```julia
RotaryPositionalEmbedding(
    dim::IntegerType;
    max_sequence_length::IntegerType=4096,
    base::IntegerType=10000,
    low_memory_variant::Bool=true,
)
```

Rotary Positional Embedding. For details see [Su *et al.* \[2\]](/references#su2024roformer).

The traditional implementation rotates consecutive pairs of elements in the feature dimension while the default implementation rotates pairs with stride half the feature dimensions for efficiency.

**Arguments**

* `dim`: The feature dimensions to be rotated. If the input feature is larger than dims then the rest is left unchanged.

**Keyword Arguments**

* `base`: The base used to compute angular frequency for each dimension in the positional encodings. Default: `10000`.

* `max_sequence_length`: The maximum sequence length. Default: `4096`.

* `low_memory_variant`: If `true` then cos and sin cache have leading dimension of `dim ÷ 2`. If `false` then cos and sin cache have leading dimension of `dim`. Default: `true`.

**Input**

* 4D `AbstractArray` such that
  * `size(x, 1) == dim`

  * `size(x, 3) ≤ max_sequence_length`

**Returns**

* 4D `AbstractArray` of the same size as the input.

**States**

* NamedTuple containing `cos_cache` and `sin_cache`.

source

```julia
SinusoidalPositionalEmbedding(
    dims::IntegerType; min_freq=0.0001f0, max_freq=1.0f0,
    scale=nothing, full_turns::Bool=false
)
```

Sinusoidal Positional Embedding. For details see [Vaswani *et al.* \[1\]](/references#vaswani2017attention).

**Arguments**

* `dims`: The dimensionality of the resulting positional embeddings.

**Keyword Arguments**

* `min_freq`: The minimum frequency expected. Default: `0.0001f0`.

* `max_freq`: The maximum frequency expected. Default: `1.0f0`.

* `scale`: A multiplicative scale for the embeddings. Default: $\sqrt{2/dims}$.

* `full_turns`: If `true` multiply the frequencies with $2\pi$. Default: `false`.

**Input**

* AbstractArray

**Returns**

* If the input array is of size `(insz...,)` then the output is of size `(dims, insz...)`.

**States**

* NamedTuple containing `sigmas`.

source

### Functional API {#Functional-API}

```julia
apply_rotary_embedding(x::AbstractArray{T,4}, cos_cache::AbstractMatrix,
                       sin_cache::AbstractMatrix; head_dim::Integer, seq_dim::Integer)
apply_rotary_embedding(x::AbstractArray{T,4}, input_positions::AbstractVector{<:Integer},
                       cos_cache::AbstractMatrix, sin_cache::AbstractMatrix;
                       head_dim::Integer, seq_dim::Integer)
```

Apply rotary embedding to the input `x` using the `cos_cache` and `sin_cache` parameters. If `input_positions` is provided, then we extract the cosine and sine cache for the corresponding positions in the sequence. Otherwise, we use the entire cache upto sequence length of `x`.

**Arguments**

* `x`: 4D `AbstractArray`.

* `cos_cache`: Cache of cosine values. Generated using [`compute_rotary_embedding_params`](/api/Lux/layers#Lux.compute_rotary_embedding_params).

* `sin_cache`: Cache of sine values. Generated using [`compute_rotary_embedding_params`](/api/Lux/layers#Lux.compute_rotary_embedding_params).

* `seq_dim`: Dimension of the sequence. Must be between 1 and 4.

* `input_positions`: Positions in the sequence to extract the cosine and sine cache for. If not provided, then we use the entire cache upto sequence length of `x`.

**Returns**

* Output of the rotary embedding.

source

```julia
compute_rotary_embedding_params(head_dim::Integer, max_sequence_length::Integer;
                                base::Number, dtype::Type{T}=Float32,
                                low_memory_variant::Bool=true)
```

Computes the cosine and sine cache for rotary positional embeddings.

**Arguments**

* `head_dim`: The feature dimensions to be rotated.

* `max_sequence_length`: The maximum sequence length. Default: `4096`.

**Keyword Arguments**

* `base`: The base used to compute angular frequency for each dimension in the positional encodings. Default: `10000`.

* `dtype`: The data type of the cache. Default: `Float32`.

* `low_memory_variant`: If `true` then cos and sin cache have leading dimension of `head_dim ÷ 2`. If `false` then cos and sin cache have leading dimension of `head_dim`. Default: `true`.

source

## Misc. Helper Layers {#Misc.-Helper-Layers}

```julia
FlattenLayer(; N = nothing)
```

Flattens the passed array into a matrix.

**Keyword Arguments**

* `N`: Flatten the first `N` dimensions of the input array. If `nothing`, then all dimensions (except the last) are flattened. Note that the batch dimension is never flattened.

**Inputs**

* `x`: AbstractArray

**Returns**

* AbstractMatrix of size `(:, size(x, ndims(x)))` if `N` is `nothing` else the first `N` dimensions of the input array are flattened.

* Empty `NamedTuple()`

**Example**

```julia
julia> model = FlattenLayer()
FlattenLayer{Nothing}(nothing)

julia> rng = Random.default_rng();
       Random.seed!(rng, 0);
       ps, st = Lux.setup(rng, model);
       x = randn(rng, Float32, (2, 2, 2, 2));

julia> y, st_new = model(x, ps, st);
       size(y)
(8, 2)
```

source

```julia
Maxout(layers...)
Maxout(; layers...)
Maxout(f::Function, n_alts::Int)
```

This contains a number of internal layers, each of which receives the same input. Its output is the elementwise maximum of the the internal layers' outputs.

Maxout over linear dense layers satisfies the universal approximation theorem \[[3](/references#goodfellow2013maxout)].

See also [`Parallel`](/api/Lux/layers#Lux.Parallel) to reduce with other operators.

**Arguments**

* Layers can be specified in three formats:
  * A list of `N` Lux layers

  * Specified as `N` keyword arguments.

  * A no argument function `f` and an integer `n_alts` which specifies the number of layers.

**Extended Help**

**Inputs**

* `x`: Input that is passed to each of the layers

**Returns**

* Output is computed by taking elementwise `max` of the outputs of the individual layers.

* Updated state of the `layers`

**Parameters**

* Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

**States**

* States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N` (naming changes if using the kwargs API)

source

```julia
NoOpLayer()
```

As the name suggests does nothing but allows pretty printing of layers. Whatever input is passed is returned.

**Example**

```julia
julia> model = NoOpLayer()
NoOpLayer()

julia> rng = Random.default_rng();
       Random.seed!(rng, 0);
       ps, st = Lux.setup(rng, model);
       x = 1
1

julia> y, st_new = model(x, ps, st)
(1, NamedTuple())
```

source

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

**Example**

```julia
julia> model = ReshapeLayer((2, 2))
ReshapeLayer(output_dims = (2, 2, :))

julia> rng = Random.default_rng();
       Random.seed!(rng, 0);
       ps, st = Lux.setup(rng, model);
       x = randn(rng, Float32, (4, 1, 3));

julia> y, st_new = model(x, ps, st);
       size(y)
(2, 2, 3)
```

source

```julia
SelectDim(dim, i)
```

Return a view of all the data of the input `x` where the index for dimension `dim` equals `i`. Equivalent to `view(x,:,:,...,i,:,:,...)` where `i` is any valid index for index slot `d` (e.g. an integer or a unit range).  Note that it may be inefficient to use non-contiguous views.

**Arguments**

* `dim`: Dimension for indexing

* `i`: Index or indices for dimension `dim`

**Inputs**

* `x`: AbstractArray that can be indexed with `view(x,:,:,...,i,:,:,...)`

**Returns**

* `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`

* Empty `NamedTuple()`

source

```julia
WrappedFunction(f)
```

Wraps a stateless and parameter less function. Might be used when a function is added to `Chain`. For example, `Chain(x -> relu.(x))` would not work and the right thing to do would be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing to do would be `Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`

**Arguments**

* `f`: Some function.

**Inputs**

* `x`: will be directly passed to `f`

**Returns**

* Output of `f(x)`

* Empty `NamedTuple()`

source

```julia
ReverseSequence(dim = nothing)
```

Reverse the specified dimension `dims` of the passed array

**Arguments**

* `dim`: Dimension that need to be reversed. If `nothing`, for AbstractVector{T} it reverses itself (dimension 1), for other arrays, reverse the dimension `ndims(x) - 1`.

**Inputs**

* `x`: AbstractArray.

**Returns**

* AbstractArray with the same dimensions as the input

* Empty `NamedTuple()`

**Example**

```julia
julia> model = ReverseSequence()
ReverseSequence{Nothing}(nothing)

julia> rng = Random.default_rng();
       Random.seed!(rng, 0);
       ps, st = Lux.setup(rng, model);
       x = [1.0, 2.0, 3.0];

julia> y, st_new = model(x, ps, st)
([3.0, 2.0, 1.0], NamedTuple())
```

source

## Normalization Layers {#Normalization-Layers}

```julia
BatchNorm(chs::Integer, activation=identity; init_bias=zeros32, init_scale=ones32,
          affine=True(), track_stats=True(), epsilon=1f-5, momentum=0.1f0,
          use_decomposed_implementation=False())
```

[Batch Normalization](https://arxiv.org/abs/1502.03167) layer.

`BatchNorm` computes the mean and variance for each $D\_1 × ... × D\_{N-2} × 1 × D\_N$ input slice and normalises the input accordingly.

**Arguments**

* `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions, call the `N-1`th the channel dimension. For a batch of feature vectors this is just the data dimension, for `WHCN` images it's the usual channel dimension.

* `activation`: After normalization, elementwise activation `activation` is applied.

**Keyword Arguments**

* If `track_stats=true`, accumulates mean and variance statistics in training phase that will be used to renormalize the input in test phase.

* `epsilon`: a value added to the denominator for numerical stability

* `momentum`:  the value used for the `running_mean` and `running_var` computation

* If `affine=true`, it also applies a shift and a rescale to the input through to learnable per-channel bias and scale parameters.
  * `init_bias`: Controls how the `bias` is initialized

  * `init_scale`: Controls how the `scale` is initialized

**Extended Help**

**Internal Keyword Arguments**

* `use_decomposed_implementation`: Several backends like CUDA, Reactant dispatch to a specialized vendored implementation for batchnorm. Setting this to true, makes Lux emit a variant of batchnorm that is computed without using any specialized kernels. This is meant to be used for correctness testing and benchmarking purposes.

**Inputs**

* `x`: Array where `size(x, N - 1) = chs`

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
julia> Chain(Dense(784 => 64), BatchNorm(64, relu), Dense(64 => 10), BatchNorm(10))
Chain(
    layer_1 = Dense(784 => 64),                   # 50_240 parameters
    layer_2 = BatchNorm(64, relu, affine=true, track_stats=true),  # 128 parameters, plus 129 non-trainable
    layer_3 = Dense(64 => 10),                    # 650 parameters
    layer_4 = BatchNorm(10, affine=true, track_stats=true),  # 20 parameters, plus 21 non-trainable
)         # Total: 51_038 parameters,
          #        plus 150 states.
```

::: warning Warning

Passing a batch size of 1, during training will result in an error.

:::

See also [`BatchNorm`](/api/Lux/layers#Lux.BatchNorm), [`InstanceNorm`](/api/Lux/layers#Lux.InstanceNorm), [`LayerNorm`](/api/Lux/layers#Lux.LayerNorm), [`WeightNorm`](/api/Lux/layers#Lux.WeightNorm)

source

```julia
GroupNorm(chs::Integer, groups::Integer, activation=identity; init_bias=zeros32,
          init_scale=ones32, affine=true, epsilon=1f-5)
```

[Group Normalization](https://arxiv.org/abs/1803.08494) layer.

**Arguments**

* `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions, call the `N-1`th the channel dimension. For a batch of feature vectors this is just the data dimension, for `WHCN` images it's the usual channel dimension.

* `groups` is the number of groups along which the statistics are computed. The number of channels must be an integer multiple of the number of groups.

* `activation`: After normalization, elementwise activation `activation` is applied.

**Keyword Arguments**

* `epsilon`: a value added to the denominator for numerical stability

* If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias and scale parameters.
  * `init_bias`: Controls how the `bias` is initialized

  * `init_scale`: Controls how the `scale` is initialized

**Extended Help**

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
julia> Chain(Dense(784 => 64), GroupNorm(64, 4, relu), Dense(64 => 10), GroupNorm(10, 5))
Chain(
    layer_1 = Dense(784 => 64),                   # 50_240 parameters
    layer_2 = GroupNorm(64, 4, relu, affine=true),  # 128 parameters
    layer_3 = Dense(64 => 10),                    # 650 parameters
    layer_4 = GroupNorm(10, 5, affine=true),      # 20 parameters
)         # Total: 51_038 parameters,
          #        plus 0 states.
```

See also [`GroupNorm`](/api/Lux/layers#Lux.GroupNorm), [`InstanceNorm`](/api/Lux/layers#Lux.InstanceNorm), [`LayerNorm`](/api/Lux/layers#Lux.LayerNorm), [`WeightNorm`](/api/Lux/layers#Lux.WeightNorm)

source

```julia
InstanceNorm(chs::Integer, activation=identity; init_bias=zeros32, init_scale=ones32,
             affine=False(), track_stats=False(), epsilon=1f-5, momentum=0.1f0)
```

Instance Normalization. For details see [Ulyanov *et al.* \[4\]](/references#ulyanov2016instance).

Instance Normalization computes the mean and variance for each $D\_1 \times ... \times D\_{N - 2} \times 1 \times 1$\` input slice and normalises the input accordingly.

**Arguments**

* `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions, call the `N-1`th the channel dimension. For a batch of feature vectors this is just the data dimension, for `WHCN` images it's the usual channel dimension.

* `activation`: After normalization, elementwise activation `activation` is applied.

**Keyword Arguments**

* If `track_stats=true`, accumulates mean and variance statistics in training phase that will be used to renormalize the input in test phase.

* `epsilon`: a value added to the denominator for numerical stability

* `momentum`:  the value used for the `running_mean` and `running_var` computation

* If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias and scale parameters.
  * `init_bias`: Controls how the `bias` is initialized

  * `init_scale`: Controls how the `scale` is initialized

**Extended Help**

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
julia> Chain(Dense(784 => 64), InstanceNorm(64, relu; affine=true), Dense(64 => 10),
           InstanceNorm(10, relu; affine=true))
Chain(
    layer_1 = Dense(784 => 64),                   # 50_240 parameters
    layer_2 = InstanceNorm(64, relu, affine=true, track_stats=false),  # 128 parameters, plus 1 non-trainable
    layer_3 = Dense(64 => 10),                    # 650 parameters
    layer_4 = InstanceNorm(10, relu, affine=true, track_stats=false),  # 20 parameters, plus 1 non-trainable
)         # Total: 51_038 parameters,
          #        plus 2 states.
```

See also [`BatchNorm`](/api/Lux/layers#Lux.BatchNorm), [`GroupNorm`](/api/Lux/layers#Lux.GroupNorm), [`LayerNorm`](/api/Lux/layers#Lux.LayerNorm), [`WeightNorm`](/api/Lux/layers#Lux.WeightNorm)

source

```julia
LayerNorm(shape::NTuple{N, Int}, activation=identity; epsilon=1f-5, dims=Colon(),
          affine=true, init_bias=zeros32, init_scale=ones32)
```

Computes mean and standard deviation over the whole input array, and uses these to normalize the whole array. Optionally applies an elementwise affine transformation afterwards.

Given an input array $x$, this layer computes

$$y = \frac{x - \mathbb{E}\[x]}{\sqrt{Var\[x] + \epsilon}} \* \gamma + \beta$$

where $\gamma$ & $\beta$ are trainable parameters if `affine=true`.

**Arguments**

* `shape`: Broadcastable shape of input array excluding the batch dimension.

* `activation`: After normalization, elementwise activation `activation` is applied.

**Keyword Arguments**

* `epsilon`: a value added to the denominator for numerical stability.

* `dims`: Dimensions to normalize the array over.

* If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-element bias and scale parameters.
  * `init_bias`: Controls how the `bias` is initialized

  * `init_scale`: Controls how the `scale` is initialized

**Extended Help**

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

source

```julia
WeightNorm(layer::AbstractLuxLayer, which_params::NTuple{N, Symbol},
           dims::Union{Tuple, Nothing}=nothing)
```

Applies [weight normalization](https://arxiv.org/abs/1602.07868) to a parameter in the given layer.

$$w = g\frac{v}{|v|}$$

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

source

```julia
RMSNorm(normalized_shape::Dims; epsilon=1.0f-5, affine=true)
RMSNorm(dim::Integer...; kwargs...)
```

Root Mean Square Normalization layer. It normalizes the input by computing the root mean square (RMS) of the first `N` dimensions of the input where `N` is the length of `normalized_shape`.

$$\begin{align}
y\_i &= \frac{x\_i}{\sqrt{\epsilon + \frac{1}{D}\Sigma\_{j=1}^D x\_j^2}} \* \gamma\_i
\end{align}$$

**Arguments**

* `normalized_shape`: The input shape from which the RMS normalization factor is computed. The input is expected to have a shape that can be broadcast with `normalized_shape`.

**Keyword Arguments**

* `epsilon`: A small value for numerical stability.

* `affine`: If `true`, learns a scale parameter.

**Extended Help**

**Inputs**

* `x`: Array of size `(normalized_shape..., *, *..., *)`

**Returns**

* `y`: Normalized Array of same shape as `x`

* Empty `NamedTuple()`

source

## Upsampling {#Upsampling}

```julia
PixelShuffle(r::Int)
```

Pixel shuffling layer with upscale factor `r`. Usually used for generating higher resolution images while upscaling them.

See `NNlib.pixel_shuffle` for more details.

**Arguments**

* `r`: Upscale factor

**Inputs**

* `x`: For 4D-arrays representing N images, the operation converts input `size(x) == (W, H, r² x C, N)` to output of size `(r x W, r x H, C, N)`. For D-dimensional data, it expects `ndims(x) == D + 2` with channel and batch dimensions, and divides the number of channels by `rᴰ`.

**Returns**

* Output of size `(r x W, r x H, C, N)` for 4D-arrays, and `(r x W, r x H, ..., C, N)` for D-dimensional data, where `D = ndims(x) - 2`

source

```julia
Upsample(mode = :nearest; [scale, size, align_corners=false])
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

**Extended Help**

**Other Keyword Arguments**

* `align_corners`: If `true`, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. This only has effect when mode is one of `:bilinear` or `:trilinear`.

**Inputs**

* `x`: For the input dimensions look into the documentation for the corresponding `NNlib` function
  * As a rule of thumb, `:nearest` should work with arrays of arbitrary dimensions

  * `:bilinear` works with 4D Arrays

  * `:trilinear` works with 5D Arrays

**Returns**

* Upsampled Input of size `size` or of size `(I_1 x scale[1], ..., I_N x scale[N], C, N)`

* Empty `NamedTuple()`

source

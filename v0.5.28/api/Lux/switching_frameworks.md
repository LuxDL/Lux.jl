
<a id='Switching-between-Deep-Learning-Frameworks'></a>

# Switching between Deep Learning Frameworks




<a id='flux-to-lux-migrate-api'></a>

## Flux Models to Lux Models


`Flux.jl` has been around in the Julia ecosystem for a long time and has a large userbase, hence we provide a way to convert `Flux` models to `Lux` models.


:::tip


Accessing these functions require manually loading `Flux`, i.e., `using Flux` must be present somewhere in the code for these to be used.


:::

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Adapt.adapt-Tuple{FromFluxAdaptor, Any}' href='#Adapt.adapt-Tuple{FromFluxAdaptor, Any}'>#</a>&nbsp;<b><u>Adapt.adapt</u></b> &mdash; <i>Method</i>.



```julia
Adapt.adapt(from::FromFluxAdaptor, L)
```

Adapt a Flux model `L` to Lux model. See [`FromFluxAdaptor`](switching_frameworks#Lux.FromFluxAdaptor) for more details.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/10dc1facb3574bf117d7de84ba48198a938ee13c/src/transform/flux.jl#L79-L83' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.FromFluxAdaptor' href='#Lux.FromFluxAdaptor'>#</a>&nbsp;<b><u>Lux.FromFluxAdaptor</u></b> &mdash; <i>Type</i>.



```julia
FromFluxAdaptor(preserve_ps_st::Bool=false, force_preserve::Bool=false)
```

Convert a Flux Model to Lux Model.

:::warning

This always ingores the `active` field of some of the Flux layers. This is almost never going to be supported.

:::

**Keyword Arguments**

  * `preserve_ps_st`: Set to `true` to preserve the states and parameters of the layer. This attempts the best possible way to preserve the original model. But it might fail. If you need to override possible failures, set `force_preserve` to `true`.
  * `force_preserve`: Some of the transformations with state and parameters preservation haven't been implemented yet, in these cases, if `force_transform` is `false` a warning will be printed and a core Lux layer will be returned. Else, it will create a [`FluxLayer`](switching_frameworks#Lux.FluxLayer).

**Example**

```julia
import Flux
using Adapt, Lux, Metalhead, Random

m = ResNet(18)
m2 = adapt(FromFluxAdaptor(), m.layers) # or FromFluxAdaptor()(m.layers)

x = randn(Float32, 224, 224, 3, 1);

ps, st = Lux.setup(Random.default_rng(), m2);

m2(x, ps, st)
```


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/10dc1facb3574bf117d7de84ba48198a938ee13c/src/transform/flux.jl#L1-L39' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.FluxLayer' href='#Lux.FluxLayer'>#</a>&nbsp;<b><u>Lux.FluxLayer</u></b> &mdash; <i>Type</i>.



```julia
FluxLayer(layer)
```

Serves as a compatibility layer between Flux and Lux. This uses `Optimisers.destructure` API internally.

:::warning

Lux was written to overcome the limitations of `destructure` + `Flux`. It is recommended to rewrite your layer in Lux instead of using this layer.

:::

:::warning

Introducing this Layer in your model will lead to type instabilities, given the way `Optimisers.destructure` works.

:::

**Arguments**

  * `layer`: Flux layer

**Parameters**

  * `p`: Flattened parameters of the `layer`


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/10dc1facb3574bf117d7de84ba48198a938ee13c/src/transform/flux.jl#L45-L72' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Lux-Models-to-Simple-Chains'></a>

## Lux Models to Simple Chains


`SimpleChains.jl` provides a way to train Small Neural Networks really fast on CPUs. See [this blog post](https://julialang.org/blog/2022/04/simple-chains/) for more details. This section describes how to convert `Lux` models to `SimpleChains` models while preserving the [layer interface](../../manual/interface#lux-interface).


:::tip


Accessing these functions require manually loading `SimpleChains`, i.e., `using SimpleChains` must be present somewhere in the code for these to be used.


:::

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Adapt.adapt-Tuple{ToSimpleChainsAdaptor, LuxCore.AbstractExplicitLayer}' href='#Adapt.adapt-Tuple{ToSimpleChainsAdaptor, LuxCore.AbstractExplicitLayer}'>#</a>&nbsp;<b><u>Adapt.adapt</u></b> &mdash; <i>Method</i>.



```julia
Adapt.adapt(from::ToSimpleChainsAdaptor, L::AbstractExplicitLayer)
```

Adapt a Flux model to Lux model. See [`ToSimpleChainsAdaptor`](switching_frameworks#Lux.ToSimpleChainsAdaptor) for more details.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/10dc1facb3574bf117d7de84ba48198a938ee13c/src/transform/simplechains.jl#L58-L62' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.ToSimpleChainsAdaptor' href='#Lux.ToSimpleChainsAdaptor'>#</a>&nbsp;<b><u>Lux.ToSimpleChainsAdaptor</u></b> &mdash; <i>Type</i>.



```julia
ToSimpleChainsAdaptor()
```

Adaptor for converting a Lux Model to SimpleChains. The returned model is still a Lux model, and satisfies the `AbstractExplicitLayer` interfacem but all internal calculations are performed using SimpleChains.

:::warning

There is no way to preserve trained parameters and states when converting to `SimpleChains.jl`.

:::

:::warning

Any kind of initialization function is not preserved when converting to `SimpleChains.jl`.

:::

**Arguments**

  * `input_dims`: Tuple of input dimensions excluding the batch dimension. These must be of `static` type as `SimpleChains` expects.

**Example**

```julia
import SimpleChains: static
using Adapt, Lux, Random

lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
    Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)))

adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))

simple_chains_model = adapt(adaptor, lux_model) # or adaptor(lux_model)

ps, st = Lux.setup(Random.default_rng(), simple_chains_model)
x = randn(Float32, 28, 28, 1, 1)

simple_chains_model(x, ps, st)
```


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/10dc1facb3574bf117d7de84ba48198a938ee13c/src/transform/simplechains.jl#L1-L45' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.SimpleChainsLayer' href='#Lux.SimpleChainsLayer'>#</a>&nbsp;<b><u>Lux.SimpleChainsLayer</u></b> &mdash; <i>Type</i>.



```julia
SimpleChainsLayer(layer)
```

Wraps a `SimpleChains` layer into a `Lux` layer. All operations are performed using `SimpleChains` but the layer satisfies the `AbstractExplicitLayer` interface.

**Arguments**

  * `layer`: SimpleChains layer


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/10dc1facb3574bf117d7de84ba48198a938ee13c/src/transform/simplechains.jl#L89-L98' class='documenter-source'>source</a><br>

</div>
<br>

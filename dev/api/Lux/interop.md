---
url: /dev/api/Lux/interop.md
---
# Interoperability between Lux and other packages {#Interoperability-between-Lux-and-other-packages}

## Switching from older frameworks {#Switching-from-older-frameworks}

### Flux Models to Lux Models {#flux-to-lux-migrate-api}

`Flux.jl` has been around in the Julia ecosystem for a long time and has a large userbase, hence we provide a way to convert `Flux` models to `Lux` models.

::: tip Tip

Accessing these functions require manually loading `Flux`, i.e., `using Flux` must be present somewhere in the code for these to be used.

:::

```julia
Adapt.adapt(from::FromFluxAdaptor, L)
```

Adapt a Flux model `L` to Lux model. See [`FromFluxAdaptor`](/api/Lux/interop#Lux.FromFluxAdaptor) for more details.

source

```julia
FromFluxAdaptor(preserve_ps_st::Bool=false, force_preserve::Bool=false)
```

Convert a Flux Model to Lux Model.

::: warning `active` field

This always ignores the `active` field of some of the Flux layers. This is almost never going to be supported.

:::

**Keyword Arguments**

* `preserve_ps_st`: Set to `true` to preserve the states and parameters of the layer. This attempts the best possible way to preserve the original model. But it might fail. If you need to override possible failures, set `force_preserve` to `true`.

* `force_preserve`: Some of the transformations with state and parameters preservation haven't been implemented yet, in these cases, if `force_transform` is `false` a warning will be printed and a core Lux layer will be returned. Else, it will create a [`FluxLayer`](/api/Lux/interop#Lux.FluxLayer).

**Example**

```julia
julia> import Flux

julia> using Adapt, Lux, Random

julia> m = Flux.Chain(Flux.Dense(2 => 3, relu), Flux.Dense(3 => 2));

julia> m2 = adapt(FromFluxAdaptor(), m); # or FromFluxAdaptor()(m.layers)

julia> x = randn(Float32, 2, 32);

julia> ps, st = Lux.setup(Random.default_rng(), m2);

julia> size(first(m2(x, ps, st)))
(2, 32)
```

source

```julia
FluxLayer(layer)
```

Serves as a compatibility layer between Flux and Lux. This uses `Optimisers.destructure` API internally.

::: warning Warning

Lux was written to overcome the limitations of `destructure` + `Flux`. It is recommended to rewrite your layer in Lux instead of using this layer.

:::

::: warning Warning

Introducing this Layer in your model will lead to type instabilities, given the way `Optimisers.destructure` works.

:::

**Arguments**

* `layer`: Flux layer

**Parameters**

* `p`: Flattened parameters of the `layer`

source

## Using a different backend for Lux {#Using-a-different-backend-for-Lux}

### Lux Models to Simple Chains {#Lux-Models-to-Simple-Chains}

`SimpleChains.jl` provides a way to train Small Neural Networks really fast on CPUs. See [this blog post](https://julialang.org/blog/2022/04/simple-chains/) for more details. This section describes how to convert `Lux` models to `SimpleChains` models while preserving the [layer interface](/manual/interface#lux-interface).

::: tip Tip

Accessing these functions require manually loading `SimpleChains`, i.e., `using SimpleChains` must be present somewhere in the code for these to be used.

:::

```julia
Adapt.adapt(from::ToSimpleChainsAdaptor, L::AbstractLuxLayer)
```

Adapt a Simple Chains model to Lux model. See [`ToSimpleChainsAdaptor`](/api/Lux/interop#Lux.ToSimpleChainsAdaptor) for more details.

source

```julia
ToSimpleChainsAdaptor(input_dims, convert_to_array::Bool=false)
```

Adaptor for converting a Lux Model to SimpleChains. The returned model is still a Lux model, and satisfies the `AbstractLuxLayer` interfacem but all internal calculations are performed using SimpleChains.

::: warning Warning

There is no way to preserve trained parameters and states when converting to `SimpleChains.jl`.

:::

::: warning Warning

Any kind of initialization function is not preserved when converting to `SimpleChains.jl`.

:::

**Arguments**

* `input_dims`: Tuple of input dimensions excluding the batch dimension. These must be of `static` type as `SimpleChains` expects.

* `convert_to_array`: SimpleChains.jl by default outputs `StrideArraysCore.StrideArray`, but this might not compose well with other packages. If `convert_to_array` is set to `true`, the output will be converted to a regular `Array`.

**Example**

```julia
julia> import SimpleChains

julia> using Adapt, Lux, Random

julia> lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
           Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
           Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)));

julia> adaptor = ToSimpleChainsAdaptor((28, 28, 1));

julia> simple_chains_model = adapt(adaptor, lux_model); # or adaptor(lux_model)

julia> ps, st = Lux.setup(Random.default_rng(), simple_chains_model);

julia> x = randn(Float32, 28, 28, 1, 1);

julia> size(first(simple_chains_model(x, ps, st)))
(10, 1)
```

source

```julia
SimpleChainsLayer(layer, to_array::Union{Bool, Val}=Val(false))
SimpleChainsLayer(layer, lux_layer, to_array)
```

Wraps a `SimpleChains` layer into a `Lux` layer. All operations are performed using `SimpleChains` but the layer satisfies the `AbstractLuxLayer` interface.

`ToArray` is a boolean flag that determines whether the output should be converted to a regular `Array` or not. Default is `false`.

**Arguments**

* `layer`: SimpleChains layer

* `lux_layer`: Potentially equivalent Lux layer that is used for printing

source

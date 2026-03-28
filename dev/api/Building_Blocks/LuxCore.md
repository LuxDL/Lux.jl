---
url: /dev/api/Building_Blocks/LuxCore.md
---
# LuxCore {#LuxCore}

`LuxCore.jl` defines the abstract layers for Lux. Allows users to be compatible with the entirely of `Lux.jl` without having such a heavy dependency. If you are depending on `Lux.jl` directly, you do not need to depend on `LuxCore.jl` (all the functionality is exported via `Lux.jl`).

## Abstract Types {#Abstract-Types}

```julia
abstract type AbstractLuxLayer
```

Abstract Type for all Lux Layers

Users implementing their custom layer, **must** implement

* `initialparameters(rng::AbstractRNG, layer::CustomAbstractLuxLayer)` – This returns a `NamedTuple` containing the trainable parameters for the layer.

* `initialstates(rng::AbstractRNG, layer::CustomAbstractLuxLayer)` – This returns a NamedTuple containing the current state for the layer. For most layers this is typically empty. Layers that would potentially contain this include `BatchNorm`, `LSTM`, `GRU`, etc.

Optionally:

* `parameterlength(layer::CustomAbstractLuxLayer)` – These can be automatically calculated, but it is recommended that the user defines these.

* `statelength(layer::CustomAbstractLuxLayer)` – These can be automatically calculated, but it is recommended that the user defines these.

See also [`AbstractLuxContainerLayer`](/api/Building_Blocks/LuxCore#LuxCore.AbstractLuxContainerLayer)

source

```julia
abstract type AbstractLuxWrapperLayer{layer} <: AbstractLuxLayer
```

See [`AbstractLuxContainerLayer`](/api/Building_Blocks/LuxCore#LuxCore.AbstractLuxContainerLayer) for detailed documentation. This abstract type is very similar to [`AbstractLuxContainerLayer`](/api/Building_Blocks/LuxCore#LuxCore.AbstractLuxContainerLayer) except that it allows for a single layer to be wrapped in a container.

Additionally, on calling [`initialparameters`](/api/Building_Blocks/LuxCore#LuxCore.initialparameters) and [`initialstates`](/api/Building_Blocks/LuxCore#LuxCore.initialstates), the parameters and states are **not** wrapped in a `NamedTuple` with the same name as the field.

As a convenience, we define the fallback call `(::AbstractLuxWrapperLayer)(x, ps, st)`, which calls `getfield(x, layer)(x, ps, st)`.

source

```julia
abstract type AbstractLuxContainerLayer{layers} <: AbstractLuxLayer
```

Abstract Container Type for certain Lux Layers. `layers` is a tuple containing fieldnames for the layer, and constructs the parameters and states using those.

Users implementing their custom layer can extend the same functions as in [`AbstractLuxLayer`](/api/Building_Blocks/LuxCore#LuxCore.AbstractLuxLayer).

::: tip Advanced Structure Manipulation

Advanced structure manipulation of these layers post construction is possible via `Functors.fmap`. For a more flexible interface, we recommend using `Lux.Experimental.@layer_map`.

:::

::: tip `fmap` Support

`fmap` support needs to be explicitly enabled by loading `Functors.jl` and `Setfield.jl`.

:::

::: warning Changes from Pre-1.0 Behavior

Previously if `layers` was a singleton tuple, [`initialparameters`](/api/Building_Blocks/LuxCore#LuxCore.initialparameters) and [`initialstates`](/api/Building_Blocks/LuxCore#LuxCore.initialstates) would return the parameters and states for the single field `layers`. From `v1.0.0` onwards, even for singleton tuples, the parameters/states are wrapped in a `NamedTuple` with the same name as the field. See [`AbstractLuxWrapperLayer`](/api/Building_Blocks/LuxCore#LuxCore.AbstractLuxWrapperLayer) to replicate the previous behavior of singleton tuples.

:::

source

## General {#General}

```julia
apply(model, x, ps, st)
```

In most cases this function simply calls `model(x, ps, st)`. However, it is still recommended to call `apply` instead of `model(x, ps, st)` directly. Some of the reasons for this include:

1. For certain types of inputs `x`, we might want to perform preprocessing before calling `model`. For eg, if `x` is an Array of `ReverseDiff.TrackedReal`s this can cause significant regressions in `model(x, ps, st)` (since it won't hit any of the BLAS dispatches). In those cases, we would automatically convert `x` to a `ReverseDiff.TrackedArray`.

2. Certain user defined inputs need to be applied to specific layers but we want the datatype of propagate through all the layers (even unsupported ones). In these cases, we can unpack the input in `apply` and pass it to the appropriate layer and then repack it before returning. See the Lux manual on Custom Input Types for a motivating example.

::: tip Tip

`apply` is integrated with `DispatchDoctor.jl` that allows automatic verification of type stability. By default this is "disable"d. For more information, see the [documentation](https://github.com/MilesCranmer/DispatchDoctor.jl).

:::

source

```julia
stateless_apply(model, x, ps)
```

Calls `apply` and only returns the first argument. This function requires that `model` has an empty state of `NamedTuple()`. Behavior of other kinds of models are undefined and it is the responsibility of the user to ensure that the model has an empty state.

source

```julia
check_fmap_condition(cond, tmatch::Union{Type, Nothing}, x) -> Bool
```

`fmap`s into the structure `x` and see if `cond` is satisfied for any of the leaf elements.

**Arguments**

* `cond` - A function that takes a single argument and returns a `Bool`.

* `tmatch` - A shortcut to check if `x` is of type `tmatch`. Can be disabled by passing `nothing`.

* `x` - The structure to check.

**Returns**

A Boolean Value

source

```julia
contains_lux_layer(l) -> Bool
```

Check if the structure `l` is a Lux AbstractLuxLayer or a container of such a layer.

source

```julia
display_name(layer::AbstractLuxLayer)
```

Printed Name of the `layer`. If the `layer` has a field `name` that is used, else the type name is used.

source

```julia
replicate(rng::AbstractRNG)
```

Creates a copy of the `rng` state depending on its type.

source

```julia
setup(rng::AbstractRNG, layer)
```

Shorthand for getting the parameters and states of the layer `l`. Is equivalent to `(initialparameters(rng, l), initialstates(rng, l))`.

::: warning Warning

This function is not pure, it mutates `rng`.

:::

source

## Parameters {#Parameters}

```julia
initialparameters(rng::AbstractRNG, layer)
```

Generate the initial parameters of the layer `l`.

source

```julia
parameterlength(layer)
```

Return the total number of parameters of the layer `l`.

source

## States {#States}

```julia
initialstates(rng::AbstractRNG, layer)
```

Generate the initial states of the layer `l`.

source

```julia
statelength(layer)
```

Return the total number of states of the layer `l`.

source

```julia
testmode(st::NamedTuple)
```

Make all occurrences of `training` in state `st` – `Val(false)`.

source

```julia
trainmode(st::NamedTuple)
```

Make all occurrences of `training` in state `st` – `Val(true)`.

source

```julia
update_state(st::NamedTuple, key::Symbol, value; exclude=Internal.isleaf)
```

Recursively update all occurrences of the `key` in the state `st` with the `value`. `exclude` is a function that is passed to `Functors.fmap_with_path`'s `exclude` keyword.

::: warning Needs Functors.jl

This function requires `Functors.jl` to be loaded.

:::

source

```julia
preserves_state_type(l::AbstractLuxLayer)
```

Return `true` if the layer `l` preserves the state type of the layer. For example, if the input state type for your layer is `T1` and the output state type is `T2`, then this function should return `T1 == T2`.

By default this function returns `true` for `AbstractLuxLayer`. For container layers, this function returns `true` if all the layers in the container preserve the state type.

source

## Layer size {#Layer-size}

```julia
outputsize(layer, x, rng)
```

Return the output size of the layer.

The fallback implementation of this function assumes the inputs were batched, i.e., if any of the outputs are Arrays, with `ndims(A) > 1`, it will return `size(A)[1:(end - 1)]`. If this behavior is undesirable, provide a custom `outputsize(layer, x, rng)` implementation).

::: warning Fallback Implementation

The fallback implementation of this function is defined once `Lux.jl` is loaded.

:::

::: warning Changes from Pre-1.0 Behavior

Previously it was possible to override this function by defining `outputsize(layer)`. However, this can potentially introduce a bug that is hard to bypass. See [this PR](https://github.com/LuxDL/LuxCore.jl/pull/43) for more information.

:::

source

## Stateful Layer {#Stateful-Layer}

```julia
StatefulLuxLayer{FT}(model, ps, st)
StatefulLuxLayer(model, ps, st)
```

::: warning Warning

This is not a LuxCore.AbstractLuxLayer

:::

::: tip Tip

This layer can be used as a drop-in replacement for `Flux.jl` layers.

:::

A convenience wrapper over Lux layers which stores the parameters and states internally. This is meant to be used in internal implementation of layers.

When using the definition of `StatefulLuxLayer` without `FT` specified, make sure that all of the layers in the model define [`LuxCore.preserves_state_type`](/api/Building_Blocks/LuxCore#LuxCore.preserves_state_type). Else we implicitly assume that the state type is preserved.

**Usecases**

* Internal implementation of `@compact` heavily uses this layer.

* In SciML codebases where propagating state might involving [`Box`ing](https://github.com/JuliaLang/julia/issues/15276). For a motivating example, see the Neural ODE tutorial.

* Facilitates Nested AD support in Lux. For more details on this feature, see the [Nested AD Manual Page](/manual/nested_autodiff#nested_autodiff).

**Static Parameters**

* If `FT = true` then the type of the `state` is fixed, i.e., `typeof(last(model(x, ps, st))) == st`.

* If `FT = false` then type of the state might change. Note that while this works in all cases, it will introduce type instability.

**Arguments**

* `model`: A Lux layer

* `ps`: The parameters of the layer. This can be set to `nothing`, if the user provides the parameters on function call

* `st`: The state of the layer

**Inputs**

* `x`: The input to the layer

* `ps`: The parameters of the layer. Optional, defaults to `s.ps`

**Outputs**

* `y`: The output of the layer

source

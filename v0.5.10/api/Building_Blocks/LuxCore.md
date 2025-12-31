
<a id='LuxCore'></a>

# LuxCore


`LuxCore.jl` defines the abstract layers for Lux. Allows users to be compatible with the entirely of `Lux.jl` without having such a heavy dependency. If you are depending on `Lux.jl` directly, you do not need to depend on `LuxCore.jl` (all the functionality is exported via `Lux.jl`).


<a id='Index'></a>

## Index

- [`LuxCore.AbstractExplicitContainerLayer`](#LuxCore.AbstractExplicitContainerLayer)
- [`LuxCore.AbstractExplicitLayer`](#LuxCore.AbstractExplicitLayer)
- [`LuxCore.apply`](#LuxCore.apply)
- [`LuxCore.check_fmap_condition`](#LuxCore.check_fmap_condition)
- [`LuxCore.contains_lux_layer`](#LuxCore.contains_lux_layer)
- [`LuxCore.display_name`](#LuxCore.display_name)
- [`LuxCore.initialparameters`](#LuxCore.initialparameters)
- [`LuxCore.initialstates`](#LuxCore.initialstates)
- [`LuxCore.parameterlength`](#LuxCore.parameterlength)
- [`LuxCore.setup`](#LuxCore.setup)
- [`LuxCore.statelength`](#LuxCore.statelength)
- [`LuxCore.testmode`](#LuxCore.testmode)
- [`LuxCore.trainmode`](#LuxCore.trainmode)
- [`LuxCore.update_state`](#LuxCore.update_state)


<a id='Abstract-Types'></a>

## Abstract Types

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.AbstractExplicitLayer' href='#LuxCore.AbstractExplicitLayer'>#</a>&nbsp;<b><u>LuxCore.AbstractExplicitLayer</u></b> &mdash; <i>Type</i>.



```julia
abstract type AbstractExplicitLayer
```

Abstract Type for all Lux Layers

Users implementing their custom layer, **must** implement

  * `initialparameters(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` – This returns a `NamedTuple` containing the trainable parameters for the layer.
  * `initialstates(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` – This returns a NamedTuple containing the current state for the layer. For most layers this is typically empty. Layers that would potentially contain this include `BatchNorm`, `LSTM`, `GRU` etc.

Optionally:

  * `parameterlength(layer::CustomAbstractExplicitLayer)` – These can be automatically calculated, but it is recommended that the user defines these.
  * `statelength(layer::CustomAbstractExplicitLayer)` – These can be automatically calculated, but it is recommended that the user defines these.

See also [`AbstractExplicitContainerLayer`](LuxCore#LuxCore.AbstractExplicitContainerLayer)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.AbstractExplicitContainerLayer' href='#LuxCore.AbstractExplicitContainerLayer'>#</a>&nbsp;<b><u>LuxCore.AbstractExplicitContainerLayer</u></b> &mdash; <i>Type</i>.



```julia
abstract type AbstractExplicitContainerLayer{layers} <: AbstractExplicitLayer
```

Abstract Container Type for certain Lux Layers. `layers` is a tuple containing fieldnames for the layer, and constructs the parameters and states using those.

Users implementing their custom layer can extend the same functions as in [`AbstractExplicitLayer`](LuxCore#LuxCore.AbstractExplicitLayer).

::: tip

Advanced structure manipulation of these layers post construction is possible via `Functors.fmap`. For a more flexible interface, we recommend using the experimental feature [`Lux.Experimental.@layer_map`](@ref).

:::

</div>
<br>

<a id='General'></a>

## General

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.apply' href='#LuxCore.apply'>#</a>&nbsp;<b><u>LuxCore.apply</u></b> &mdash; <i>Function</i>.



```julia
apply(model, x, ps, st)
```

Simply calls `model(x, ps, st)`

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.check_fmap_condition' href='#LuxCore.check_fmap_condition'>#</a>&nbsp;<b><u>LuxCore.check_fmap_condition</u></b> &mdash; <i>Function</i>.



```julia
check_fmap_condition(cond, tmatch, x) -> Bool
```

`fmap`s into the structure `x` and see if `cond` is statisfied for any of the leaf elements.

**Arguments**

```
* `cond` - A function that takes a single argument and returns a `Bool`.
* `tmatch` - A shortcut to check if `x` is of type `tmatch`. Can be disabled by passing
  `nothing`.
* `x` - The structure to check.
```

**Returns**

A Boolean Value

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.contains_lux_layer' href='#LuxCore.contains_lux_layer'>#</a>&nbsp;<b><u>LuxCore.contains_lux_layer</u></b> &mdash; <i>Function</i>.



```julia
contains_lux_layer(l) -> Bool
```

Check if the structure `l` is a Lux AbstractExplicitLayer or a container of such a layer.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.display_name' href='#LuxCore.display_name'>#</a>&nbsp;<b><u>LuxCore.display_name</u></b> &mdash; <i>Function</i>.



```julia
display_name(layer::AbstractExplicitLayer)
```

Printed Name of the `layer`. If the `layer` has a field `name` that is used, else the type name is used.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.setup' href='#LuxCore.setup'>#</a>&nbsp;<b><u>LuxCore.setup</u></b> &mdash; <i>Function</i>.



```julia
setup(rng::AbstractRNG, layer)
```

Shorthand for getting the parameters and states of the layer `l`. Is equivalent to `(initialparameters(rng, l), initialstates(rng, l))`.

::: warning

This function is not pure, it mutates `rng`.

:::

</div>
<br>

<a id='Parameters'></a>

## Parameters

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.initialparameters' href='#LuxCore.initialparameters'>#</a>&nbsp;<b><u>LuxCore.initialparameters</u></b> &mdash; <i>Function</i>.



```julia
initialparameters(rng::AbstractRNG, layer)
```

Generate the initial parameters of the layer `l`.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.parameterlength' href='#LuxCore.parameterlength'>#</a>&nbsp;<b><u>LuxCore.parameterlength</u></b> &mdash; <i>Function</i>.



```julia
parameterlength(layer)
```

Return the total number of parameters of the layer `l`.

</div>
<br>

<a id='States'></a>

## States

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.initialstates' href='#LuxCore.initialstates'>#</a>&nbsp;<b><u>LuxCore.initialstates</u></b> &mdash; <i>Function</i>.



```julia
initialstates(rng::AbstractRNG, layer)
```

Generate the initial states of the layer `l`.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.statelength' href='#LuxCore.statelength'>#</a>&nbsp;<b><u>LuxCore.statelength</u></b> &mdash; <i>Function</i>.



```julia
statelength(layer)
```

Return the total number of states of the layer `l`.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.testmode' href='#LuxCore.testmode'>#</a>&nbsp;<b><u>LuxCore.testmode</u></b> &mdash; <i>Function</i>.



```julia
testmode(st::NamedTuple)
```

Make all occurances of `training` in state `st` – `Val(false)`.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.trainmode' href='#LuxCore.trainmode'>#</a>&nbsp;<b><u>LuxCore.trainmode</u></b> &mdash; <i>Function</i>.



```julia
trainmode(st::NamedTuple)
```

Make all occurances of `training` in state `st` – `Val(true)`.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxCore.update_state' href='#LuxCore.update_state'>#</a>&nbsp;<b><u>LuxCore.update_state</u></b> &mdash; <i>Function</i>.



```julia
update_state(st::NamedTuple, key::Symbol, value;
    layer_check=_default_layer_check(key))
```

Recursively update all occurances of the `key` in the state `st` with the `value`.

</div>
<br>

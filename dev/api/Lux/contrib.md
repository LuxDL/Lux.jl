---
url: /dev/api/Lux/contrib.md
---
# Experimental Features {#Experimental-Features}

All features listed on this page are **experimental** which means:

1. No SemVer Guarantees. We use code here to iterate fast. That said, historically we have never broken any code in this module and have always provided a deprecation period.

2. Expect edge-cases and report them. It will help us move these features out of experimental sooner.

3. None of the features are exported.

## Parameter Freezing {#Parameter-Freezing}

```julia
FrozenLayer(l::AbstractLuxLayer, which_params::Optional{Tuple})
```

Freeze the parameters with name `which_params` of the layer `l`.

::: tip Use `Lux.Experimental.freeze` instead

It is always recommended to use the [`Lux.Experimental.freeze`](/api/Lux/contrib#Lux.Experimental.freeze) function instead of directly using the `FrozenLayer` constructor.

:::

::: warning No checks for `which_params`

There are no checks for `which_params`. For example, if the original layer has parameters named `(:weight, :bias)`, and `which_params` is set to `(:myweight,)` then none of the parameters are frozen and no error is thrown.

:::

**Arguments**

* `l`: Lux AbstractLuxLayer.

* `which_params`: Parameter Names to be Frozen. Can be set to `nothing`, in which case all parameters are frozen.

**Extended Help**

**Parameters**

* Parameters of the layer `l` excluding `which_params`.

**States**

* `frozen_params`: Parameters that are frozen, i.e., `which_params`.

* `states`: The state of the inner layer `l`.

**Note on Internal Layer Implementation**

The inner layer should work with `NamedTuple` parameters. In order to support custom parameter types, users need to implement `Lux.Utils.merge(::CustomParamType, ::NamedTuple)` or extend `Lux.Utils.named_tuple(::CustomParamType)` to return a `NamedTuple`.

**Example**

```julia
julia> Lux.Experimental.FrozenLayer(Dense(2 => 2), (:weight,))
FrozenLayer(Dense(2 => 2), (:weight,))  # 2 parameters, plus 4 non-trainable
```

See also [`Lux.Experimental.freeze`](/api/Lux/contrib#Lux.Experimental.freeze), [`Lux.Experimental.unfreeze`](/api/Lux/contrib#Lux.Experimental.unfreeze).

source

```julia
freeze(l::AbstractLuxLayer, which_params::Optional{Tuple} = nothing)
```

Constructs a version of `l` with `which_params` frozen. If `which_params` is nothing, then all parameters are frozen.

source

```julia
freeze(l::AbstractLuxLayer, ps, st::NamedTuple,
    which_params::Optional{Tuple} = nothing)
```

Construct a [`Lux.Experimental.FrozenLayer`](/api/Lux/contrib#Lux.Experimental.FrozenLayer) for `l` with the current parameters and states. If `which_params` is nothing, then all parameters are frozen.

source

```julia
unfreeze(l::FrozenLayer)
```

Unfreezes the layer `l`.

source

```julia
unfreeze(l::FrozenLayer, ps, st::NamedTuple)
```

Unwraps a [`Lux.Experimental.FrozenLayer`](/api/Lux/contrib#Lux.Experimental.FrozenLayer) `l` with the current parameters and states.

source

For detailed usage example look at the [manual page](/manual/freezing_model_parameters#freezing-model-parameters).

## Map over Layer {#Map-over-Layer}

```julia
layer_map(f, l::AbstractLuxLayer, ps, st::NamedTuple)
```

Map the function `f` over the model `l`, with the parameters `ps` and states `st`. This is different from `Functors.fmap` since it zips the layers, parameters, and states and invokes the function on all of them together.

::: tip KeyPath provided to the function

The `KeyPath` depths on the structure of the parameters and states. This is of consequence exclusively for [`AbstractLuxWrapperLayer`](/api/Building_Blocks/LuxCore#LuxCore.AbstractLuxWrapperLayer) where the structure of the layer doesn't match the structure of the parameters and states. In the example, provided below, the `KeyPath` is `(:chain, :dense_1)` for the first layer (following the structure in `ps`) while accessing the same layer in the chain is done with `(  :chain, :layers, :dense_1)`.

:::

**Call Signature for `f`**

* Must take 4 inputs – `AbstractLuxLayer`, Corresponding Parameters, Corresponding States, and the `Functors.KeyPath` to the layer.

* Must return a tuple of 3 elements – `AbstractLuxLayer`, new parameters and the new states.

**Extended Help**

**Example**

```julia
julia> using Lux, Random

julia> c = Parallel(
           +; chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3), dense_2=Dense(3 => 5)),
           dense_3=Dense(5 => 1));

julia> rng = Random.default_rng();

julia> ps, st = Lux.setup(rng, c);

julia> # Makes parameters of Dense Layers inside Chain zero
       function zero_dense_params(l, ps, st, name)
           if l isa Dense
               println("zeroing params of $name")
               ps = merge(ps, (; weight=zero.(ps.weight), bias=zero.(ps.bias)))
           end
           return l, ps, st
       end;

julia> _, ps_new, _ = Lux.Experimental.layer_map(zero_dense_params, c, ps, st);
zeroing params of KeyPath(:chain, :dense_1)
zeroing params of KeyPath(:chain, :dense_2)
zeroing params of KeyPath(:dense_3,)

julia> all(iszero, (ps_new.chain.dense_1.weight, ps_new.chain.dense_1.bias,
                    ps_new.chain.dense_2.weight, ps_new.chain.dense_2.bias,
                    ps_new.dense_3.weight, ps_new.dense_3.bias))
true
```

source

## Debugging Functionality {#Debugging-Functionality}

Model not working properly! Here are some functionalities to help you debug you Lux model.

```julia
@debug_mode layer kwargs...
```

Recurses into the `layer` and replaces the inner most non Container Layers with a [`Lux.Experimental.DebugLayer`](/api/Lux/contrib#Lux.Experimental.DebugLayer).

See [`Lux.Experimental.DebugLayer`](/api/Lux/contrib#Lux.Experimental.DebugLayer) for details about the Keyword Arguments.

source

```julia
DebugLayer(layer::AbstractLuxLayer;
    nan_check::Union{Symbol, StaticSymbol, Val}=static(:both),
    error_check::Union{StaticBool, Bool, Val{true}, Val{false}}=True(),
    location::KeyPath=KeyPath())
```

A wrapper over Lux layers that adds checks for NaNs and errors. This is useful for debugging.

**Arguments**

* `layer`: The layer to be wrapped.

**Extended Help**

**Keyword Arguments**

* `nan_check`: Whether to check for NaNs in the input, parameters, and states. Can be `:both`, `:forward`, `:backward`, or `:none`.

* `error_check`: Whether to check for errors in the layer. If `true`, will throw an error if the layer fails.

* `location`: The location of the layer. Use [`Lux.Experimental.@debug_mode`](/api/Lux/contrib#Lux.Experimental.@debug_mode) to construct this layer to populate this value correctly.

**Input / Output**

Inputs and outputs are the same as the `layer` unless one of the `nan_check` or `error_check` criteria is met.

If `nan_check` is enabled and NaNs are detected then a `DomainError` is thrown. If `error_check` is enabled, then any errors in the layer are thrown with useful information to track where the error originates.

::: warning ChainRules Compatible Reverse Mode AD Tools

`nan_check` for the backward mode only works with ChainRules Compatible Reverse Mode AD Tools currently.

:::

::: danger Disable After Debugging

This layer is only meant to be used for debugging. If used for actual training or inference, will lead to extremely bad performance.

:::

See [`Lux.Experimental.@debug_mode`](/api/Lux/contrib#Lux.Experimental.@debug_mode) to construct this layer.

source

## Tied Parameters {#Tied-Parameters}

```julia
share_parameters(ps, sharing)
share_parameters(ps, sharing, new_parameters)
```

Updates the parameters in `ps` with a common set of parameters `new_parameters` that are shared between each list in the nested list `sharing`. (That was kind of a mouthful, the example should make it clear).

**Arguments**

* `ps`: Original parameters.

* `sharing`: A nested list of lists of accessors of `ps` which need to shate the parameters (See the example for details). (Each list in the list must be disjoint)

* `new_parameters`: If passed the length of `new_parameters` must be equal to the length of `sharing`. For each vector in `sharing` the corresponding parameter in `new_parameters` will be used. (If not passed, the parameters corresponding to the first element of each vector in `sharing` will be used).

**Returns**

Updated Parameters having the same structure as `ps`.

**Example**

```julia
julia> model = Chain(; d1=Dense(2 => 4, tanh),
           d3=Chain(; l1=Dense(4 => 2), l2=Dense(2 => 4)), d2=Dense(4 => 2))
Chain(
    d1 = Dense(2 => 4, tanh),                     # 12 parameters
    d3 = Chain(
        l1 = Dense(4 => 2),                       # 10 parameters
        l2 = Dense(2 => 4),                       # 12 parameters
    ),
    d2 = Dense(4 => 2),                           # 10 parameters
)         # Total: 44 parameters,
          #        plus 0 states.

julia> ps, st = Lux.setup(Xoshiro(0), model);

julia> # share parameters of (d1 and d3.l1) and (d3.l2 and d2)
       ps = Lux.Experimental.share_parameters(ps, (("d3.l2", "d1"), ("d2", "d3.l1")));

julia> ps.d3.l2.weight === ps.d1.weight &&
           ps.d3.l2.bias === ps.d1.bias &&
           ps.d2.weight === ps.d3.l1.weight &&
           ps.d2.bias === ps.d3.l1.bias
true
```

::: danger ComponentArrays

ComponentArrays doesn't allow sharing parameters. Converting the returned parameters to a ComponentArray will silently cause the parameter sharing to be undone.

:::

source


<a id='Experimental-Features'></a>

# Experimental Features




All features listed on this page are **experimental** which means:


1. No SemVer Guarantees. We use code here to iterate fast and most users should wait for these features to be marked non-experimental.
2. The code will probably be moved into a separate repository in the future.
3. Expect edge-cases and report them. It will help us move these features out of experimental sooner.
4. None of the features are exported.


:::warning


Starting v"0.5.2" all Experimental features need to be accessed via `Lux.Experimental.<feature>`. Direct access via `Lux.<feature>` will be removed in v"0.6".


:::


<a id='Index'></a>

## Index

- [`Lux.Experimental.DebugLayer`](#Lux.Experimental.DebugLayer)
- [`Lux.Experimental.FrozenLayer`](#Lux.Experimental.FrozenLayer)
- [`Lux.Experimental.StatefulLuxLayer`](#Lux.Experimental.StatefulLuxLayer)
- [`Lux.Experimental.TrainState`](#Lux.Experimental.TrainState)
- [`Lux.Experimental.apply_gradients`](#Lux.Experimental.apply_gradients)
- [`Lux.Experimental.compute_gradients`](#Lux.Experimental.compute_gradients)
- [`Lux.Experimental.freeze`](#Lux.Experimental.freeze)
- [`Lux.Experimental.layer_map`](#Lux.Experimental.layer_map)
- [`Lux.Experimental.share_parameters`](#Lux.Experimental.share_parameters)
- [`Lux.Experimental.unfreeze`](#Lux.Experimental.unfreeze)
- [`Lux.Experimental.@compact`](#Lux.Experimental.@compact)
- [`Lux.Experimental.@debug_mode`](#Lux.Experimental.@debug_mode)
- [`Lux.Experimental.@layer_map`](#Lux.Experimental.@layer_map)


<a id='Training'></a>

## Training


Helper Functions making it easier to train `Lux.jl` models.


Lux.Training is meant to be simple and provide extremely basic functionality. We provide basic building blocks which can be seamlessly composed to create complex training pipelines.

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.TrainState' href='#Lux.Experimental.TrainState'>#</a>&nbsp;<b><u>Lux.Experimental.TrainState</u></b> &mdash; <i>Type</i>.



```julia
TrainState
```

Training State containing:

  * `model`: `Lux` model.
  * `parameters`: Trainable Variables of the `model`.
  * `states`: Non-trainable Variables of the `model`.
  * `optimizer_state`: Optimizer State.
  * `step`: Number of updates of the parameters made.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/training.jl#L3-L13' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.compute_gradients' href='#Lux.Experimental.compute_gradients'>#</a>&nbsp;<b><u>Lux.Experimental.compute_gradients</u></b> &mdash; <i>Function</i>.



```julia
compute_gradients(ad::ADTypes.AbstractADType, objective_function::Function, data,
    ts::TrainState)
```

Compute the gradients of the objective function wrt parameters stored in `ts`.

**Arguments**

  * `ad`: Backend (from [ADTypes.jl](https://github.com/SciML/ADTypes.jl)) used to compute the gradients.
  * `objective_function`: Objective function. The function must take 4 inputs – model, parameters, states and data. The function must return 3 values – loss, updated_state, and any computed statistics.
  * `data`: Data used to compute the gradients.
  * `ts`: Current Training State. See [`TrainState`](contrib#Lux.Experimental.TrainState).

**Return**

A 4-Tuple containing:

  * `grads`: Computed Gradients.
  * `loss`: Loss from the objective function.
  * `stats`: Any computed statistics from the objective function.
  * `ts`: Updated Training State.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/training.jl#L68-L92' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.apply_gradients' href='#Lux.Experimental.apply_gradients'>#</a>&nbsp;<b><u>Lux.Experimental.apply_gradients</u></b> &mdash; <i>Function</i>.



```julia
apply_gradients(ts::TrainState, grads)
```

Update the parameters stored in `ts` using the gradients `grads`.

**Arguments**

  * `ts`: `TrainState` object.
  * `grads`: Gradients of the loss function wrt `ts.params`.

**Returns**

Updated `TrainState` object.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/training.jl#L49-L62' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Parameter-Freezing'></a>

## Parameter Freezing


:::info


In the long term, this will be supported via [Optimisers.jl](https://github.com/FluxML/Optimisers.jl/pull/49).


:::

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.FrozenLayer' href='#Lux.Experimental.FrozenLayer'>#</a>&nbsp;<b><u>Lux.Experimental.FrozenLayer</u></b> &mdash; <i>Type</i>.



```julia
FrozenLayer(l::AbstractExplicitLayer, which_params::Union{Tuple, Nothing})
```

Freeze the parameters with name `which_params` of the layer `l`.

:::tip

It is always recommended to use the [`Lux.Experimental.freeze`](contrib#Lux.Experimental.freeze) function instead of directly using the `FrozenLayer` constructor.

:::

:::warning

There are no checks for `which_params`. For example, if the original layer has parameters named `(:weight, :bias)`, and `which_params`is set to`(:myweight,)` then none of the parameters are frozen and no error is thrown.

:::

**Arguments**

  * `l`: Lux AbstractExplicitLayer.
  * `which_params`: Parameter Names to be Frozen. Can be set to `nothing`, in which case all parameters are frozen.

**Input**

  * `x`: Input to the layer `l`.

**Returns**

  * Output of the inner layer `l`
  * Updated State

**Parameters**

  * Parameters of the layer `l` excluding `which_params`.

**States**

  * `frozen_params`: Parameters that are frozen, i.e., `which_params`.
  * `states`: The state of the inner layer `l`.

**Note on Internal Layer Implementation**

The inner layer should work with `NamedTuple` parameters. In order to support custom parameter types, users need to implement `Lux._merge(::CustomParamType, ::NamedTuple)`.

**Example**

```julia
m = Lux.Experimental.FrozenLayer(Dense(2 => 2), (:weight,))
```

See also [`Lux.Experimental.freeze`](contrib#Lux.Experimental.freeze), [`Lux.Experimental.unfreeze`](contrib#Lux.Experimental.unfreeze).


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/freeze.jl#L1-L57' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.freeze' href='#Lux.Experimental.freeze'>#</a>&nbsp;<b><u>Lux.Experimental.freeze</u></b> &mdash; <i>Function</i>.



```julia
freeze(l::AbstractExplicitLayer, which_params::Union{Tuple, Nothing} = nothing)
```

Constructs a version of `l` with `which_params` frozen. If `which_params` is nothing, then all parameters are frozen.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/freeze.jl#L107-L112' class='documenter-source'>source</a><br>


```
freeze(l::AbstractExplicitLayer, ps, st::NamedTuple,
       which_params::Union{Tuple, Nothing} = nothing)
```

Construct a [`Lux.Experimental.FrozenLayer`](contrib#Lux.Experimental.FrozenLayer) for `l` with the current parameters and states. If `which_params` is nothing, then all parameters are frozen.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/freeze.jl#L117-L123' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.unfreeze' href='#Lux.Experimental.unfreeze'>#</a>&nbsp;<b><u>Lux.Experimental.unfreeze</u></b> &mdash; <i>Function</i>.



```julia
unfreeze(l::FrozenLayer)
```

Unfreezes the layer `l`.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/freeze.jl#L141-L145' class='documenter-source'>source</a><br>


```
unfreeze(l::FrozenLayer, ps, st::NamedTuple)
```

Unwraps a [`Lux.Experimental.FrozenLayer`](contrib#Lux.Experimental.FrozenLayer) `l` with the current parameters and states.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/freeze.jl#L148-L152' class='documenter-source'>source</a><br>

</div>
<br>

For detailed usage example look at the [manual page](../../manual/freezing_model_parameters).


<a id='Map-over-Layer'></a>

## Map over Layer

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.layer_map' href='#Lux.Experimental.layer_map'>#</a>&nbsp;<b><u>Lux.Experimental.layer_map</u></b> &mdash; <i>Function</i>.



```julia
layer_map(f::Function, l::AbstractExplicitLayer, ps, st::NamedTuple,
          name::String="model")
```

Map the function `f` over the model `l`, with the parameters `ps` and states `st`. This is different from `Functors.fmap` since it zips the layers, parameters, and states and invokes the function on all of them together.

**Call Signature for `f`**

  * Must take 4 inputs – `AbstractExplicitLayer`, Corresponding Parameters, Corresponding States, and the name of the layer.
  * Must return a tuple of 3 elements – `AbstractExplicitLayer`, new parameters and the new states.

:::tip

We recommend using the macro `Lux.@layer_map` instead of this function. It automatically sets the `name` of the layer to be the variable name.

:::

**Example**

```julia
using Lux, Random, Setfield

c = Parallel(+; chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3),
                              dense_2=Dense(3 => 5)),
             dense_3=Dense(5 => 1))

rng = Random.default_rng()
ps, st = Lux.setup(rng, c)

# Makes parameters of Dense Layers inside Chain zero
function zero_dense_params(l, ps, st, name)
    if l isa Dense
        println("zeroing params of $name")
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

Lux.layer_map(zero_dense_params, c, ps, st)
```


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/map.jl#L42' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.@layer_map' href='#Lux.Experimental.@layer_map'>#</a>&nbsp;<b><u>Lux.Experimental.@layer_map</u></b> &mdash; <i>Macro</i>.



```julia
@layer_map func layer ps st
```

See the documentation of [`Lux.Experimental.layer_map`](contrib#Lux.Experimental.layer_map) for more details. This macro eliminates the need to the set the layer name, and uses the variable name as the starting point.

**Example**

```julia
using Lux, Random, Setfield

c = Parallel(+; chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3),
                              dense_2=Dense(3 => 5)),
             dense_3=Dense(5 => 1))

rng = Random.default_rng()
ps, st = Lux.setup(rng, c)

# Makes parameters of Dense Layers inside Chain zero
function zero_dense_params(l, ps, st, name)
    if l isa Dense
        println("zeroing params of $name")
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

Lux.@layer_map zero_dense_params c ps st
```


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/map.jl#L4' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Debugging-Functionality'></a>

## Debugging Functionality


Model not working properly! Here are some functionalities to help you debug you Lux model.

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.@debug_mode' href='#Lux.Experimental.@debug_mode'>#</a>&nbsp;<b><u>Lux.Experimental.@debug_mode</u></b> &mdash; <i>Macro</i>.



```julia
@debug_mode layer kwargs...
```

Recurses into the `layer` and replaces the inner most non Container Layers with a [`Lux.Experimental.DebugLayer`](contrib#Lux.Experimental.DebugLayer).

See [`Lux.Experimental.DebugLayer`](contrib#Lux.Experimental.DebugLayer) for details about the Keyword Arguments.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/debug.jl#L158-L165' class='documenter-source'>source</a><br>

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.DebugLayer' href='#Lux.Experimental.DebugLayer'>#</a>&nbsp;<b><u>Lux.Experimental.DebugLayer</u></b> &mdash; <i>Type</i>.



```julia
DebugLayer(layer::AbstractExplicitLayer; nan_check::Symbol=:both,
    error_check::Bool=true, location::String="")
```

::: danger

This layer is only meant to be used for debugging. If used for actual training or inference, will lead to extremely bad performance.

:::

A wrapper over Lux layers that adds checks for NaNs and errors. This is useful for debugging.

**Arguments**

  * `layer`: The layer to be wrapped.

**Keyword Arguments**

  * `nan_check`: Whether to check for NaNs in the input, parameters, and states. Can be `:both`, `:forward`, `:backward`, or `:none`.
  * `error_check`: Whether to check for errors in the layer. If `true`, will throw an error if the layer fails.
  * `location`: The location of the layer. Use [`Lux.Experimental.@debug_mode`](contrib#Lux.Experimental.@debug_mode) to construct this layer to populate this value correctly.

**Inputs**

  * `x`: The input to the layer.

**Outputs**

  * `y`: The output of the layer.
  * `st`: The updated states of the layer.

If `nan_check` is enabled and NaNs are detected then a `DomainError` is thrown. If `error_check` is enabled, then any errors in the layer are thrown with useful information to track where the error originates.

::: warning

`nan_check` for the backward mode only works with ChainRules Compatible Reverse Mode AD Tools currently.

:::

See [`Lux.Experimental.@debug_mode`](contrib#Lux.Experimental.@debug_mode) to construct this layer.


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/debug.jl#L1-L49' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Tied-Parameters'></a>

## Tied Parameters

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.share_parameters' href='#Lux.Experimental.share_parameters'>#</a>&nbsp;<b><u>Lux.Experimental.share_parameters</u></b> &mdash; <i>Function</i>.



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
model = Chain(;
    d1=Dense(2 => 4, tanh),
    d3=Chain(; l1=Dense(4 => 2), l2=Dense(2 => 4)),
    d2=Dense(4 => 2))

ps, st = Lux.setup(Xoshiro(0), model)

# share parameters of (d1 and d3.l1) and (d3.l2 and d2)
ps = Lux.share_parameters(ps, (("d3.l2", "d1"), ("d2", "d3.l1")))
```


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/share_parameters.jl#L3-L38' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Stateful-Layer'></a>

## Stateful Layer

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.StatefulLuxLayer' href='#Lux.Experimental.StatefulLuxLayer'>#</a>&nbsp;<b><u>Lux.Experimental.StatefulLuxLayer</u></b> &mdash; <i>Type</i>.



```julia
StatefulLuxLayer(model, ps, st)
```

::: warning

This is not a Lux.AbstractExplicitLayer

:::

A convenience wrapper over Lux layers which stores the parameters and states internally. Most users should not be using this version. This comes handy when Lux internally uses the `@compact` to construct models and in SciML codebases where propagating state might involving [`Box`ing](https://github.com/JuliaLang/julia/issues/15276).

For a motivating example, see the Neural ODE tutorial.

::: warning

State is mutated in place. An additional caveat is that the updated state from `model` must have the same type as `st`.

:::

**Arguments**

  * `model`: A Lux layer
  * `ps`: The parameters of the layer. This can be set to `nothing`, if the user provides the parameters on function call
  * `st`: The state of the layer

**Inputs**

  * `x`: The input to the layer
  * `ps`: The parameters of the layer. Optional, defaults to `s.ps`

**Outputs**

  * `y`: The output of the layer


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/stateful.jl#L1-L39' class='documenter-source'>source</a><br>

</div>
<br>

<a id='Compact-Layer-API'></a>

## Compact Layer API

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Lux.Experimental.@compact' href='#Lux.Experimental.@compact'>#</a>&nbsp;<b><u>Lux.Experimental.@compact</u></b> &mdash; <i>Macro</i>.



```julia
@compact(kw...) do x
    ...
end
@compact(forward::Function; name=nothing, parameters...)
```

Creates a layer by specifying some `parameters`, in the form of keywords, and (usually as a `do` block) a function for the forward pass. You may think of `@compact` as a specialized `let` block creating local variables that are trainable in Lux. Declared variable names may be used within the body of the `forward` function. Note that unlike typical Lux models, the forward function doesn't need to explicitly manage states.

Here is a linear model:

```julia
using Lux, Random
import Lux.Experimental: @compact

r = @compact(w=rand(3)) do x
    return w .* x
end
ps, st = Lux.setup(Xoshiro(0), r)
r([1, 1, 1], ps, st)  # x is set to [1, 1, 1].
```

Here is a linear model with bias and activation:

```julia
d_in = 5
d_out = 7
d = @compact(W=randn(d_out, d_in), b=zeros(d_out), act=relu) do x
    y = W * x
    return act.(y .+ b)
end
ps, st = Lux.setup(Xoshiro(0), d)
d(ones(5, 10), ps, st) # 7×10 Matrix as output.

ps_dense = (; weight=ps.W, bias=ps.b)
first(d([1, 2, 3, 4, 5], ps, st)) ≈
first(Dense(d_in => d_out, relu)([1, 2, 3, 4, 5], ps_dense, NamedTuple())) # Equivalent to a dense layer
```

Finally, here is a simple MLP:

```julia
n_in = 1
n_out = 1
nlayers = 3

model = @compact(w1=Dense(n_in, 128),
    w2=[Dense(128, 128) for i in 1:nlayers],
    w3=Dense(128, n_out),
    act=relu) do x
    embed = act(w1(x))
    for w in w2
        embed = act(w(embed))
    end
    out = w3(embed)
    return out
end

ps, st = Lux.setup(Xoshiro(0), model)

model(randn(n_in, 32), ps, st)  # 1×32 Matrix as output.
```

We can train this model just like any Lux model:

```julia
using Optimisers, Zygote

x_data = collect(-2.0f0:0.1f0:2.0f0)'
y_data = 2 .* x_data .- x_data .^ 3
optim = Optimisers.setup(Adam(), ps)

for epoch in 1:1000
    loss, gs = Zygote.withgradient(ps -> sum(abs2, first(model(x_data, ps, st)) .- y_data),
        ps)
    @show epoch, loss
    Optimisers.update!(optim, ps, gs[1])
end
```

You may also specify a `name` for the model, which will be used instead of the default printout, which gives a verbatim representation of the code used to construct the model:

```julia
model = @compact(w=rand(3), name="Linear(3 => 1)") do x
    return sum(w .* x)
end

println(model)  # "Linear(3 => 1)()"
```

This can be useful when using `@compact` to hierarchically construct complex models to be used inside a `Chain`.

:::tip Type Stability

If your input function `f` is type-stable but the generated model is not type stable, it should be treated as a bug. We will appreciate issues if you find such cases.

:::

:::warning Parameter Count

Array Parameter don't print the number of parameters on the side. However, they do account for the total number of parameters printed at the bottom.

:::


<a target='_blank' href='https://github.com/LuxDL/Lux.jl/blob/793ee5b25eb6d8d4aae0e73e0c3fe84c3948e024/src/contrib/compact.jl#L16-L126' class='documenter-source'>source</a><br>

</div>
<br>

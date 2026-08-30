---
url: /dev/manual/interface.md
---
# Lux Interface {#lux-interface}

::: tip Lux.jl vs LuxCore.jl

If you just want to define compatibility with Lux without actually using any of the other functionality provided by Lux (like layers), it is recommended to depend on `LuxCore.jl` instead of `Lux.jl`. `LuxCore.jl` is a significantly lighter dependency.

:::

Following this interface provides the ability for frameworks built on top of Lux to be cross compatible. Additionally, any new functionality built into Lux, will just work for your framework.

::: tip `@compact` macro

While writing out a custom struct and defining dispatches manually is a good way to understand the interface, it is not the most concise way. We recommend using the [`Lux.@compact`](/api/Lux/utilities#Lux.@compact) macro to define layers which makes handling the states and parameters downright trivial.

:::

## Layer Interface {#Layer-Interface}

### Singular Layer {#Singular-Layer}

If the layer doesn't contain any other Lux layer, then it is a `Singular Layer`. This means it should optionally subtype `Lux.AbstractLuxLayer` but mandatorily define all the necessary functions mentioned in the docstrings. Consider a simplified version of [`Dense`](/api/Lux/layers#Lux.Dense) called `Linear`.

First, setup the architectural details for this layer. Note, that the architecture doesn't contain any mutable structure like arrays. When in doubt, remember, once constructed a model architecture cannot change.

::: tip Tip

For people coming from Flux.jl background, this might be weird. We recommend checking out [the Flux to Lux migration guide](/manual/migrate_from_flux#migrate-from-flux) first before proceeding.

:::

```julia
using LuxCore, Random, WeightInitializers # Importing `Lux` also gives you access to `LuxCore`

struct Linear{F1, F2} <: LuxCore.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    init_weight::F1
    init_bias::F2
end

function Linear(in_dims::Int, out_dims::Int; init_weight=glorot_uniform, init_bias=zeros32)
    return Linear{typeof(init_weight), typeof(init_bias)}(in_dims, out_dims, init_weight,
        init_bias)
end

l = Linear(2, 4)
```

```ansi
Main.Linear{typeof(glorot_uniform), typeof(zeros32)}(2, 4, WeightInitializers.glorot_uniform, WeightInitializers.zeros32)
```

Next, we need to implement functions which return the parameters and states for the layer. In case of `Linear`, the parameters are `weight` and `bias` while the states are empty. States become important when defining layers like [`BatchNorm`](/api/Lux/layers#Lux.BatchNorm), [`WeightNorm`](/api/Lux/layers#Lux.WeightNorm), etc. The recommended data structure for returning parameters is a NamedTuple, though anything satisfying the [Parameter Interface](/manual/interface#parameter-interface) is valid.

```julia
function LuxCore.initialparameters(rng::AbstractRNG, l::Linear)
    return (weight=l.init_weight(rng, l.out_dims, l.in_dims),
            bias=l.init_bias(rng, l.out_dims, 1))
end

LuxCore.initialstates(::AbstractRNG, ::Linear) = NamedTuple()
```

You could also implement `LuxCore.parameterlength` and `LuxCore.statelength` to prevent wasteful reconstruction of the parameters and states.

```julia
# This works
println("Parameter Length: ", LuxCore.parameterlength(l), "; State Length: ",
    LuxCore.statelength(l))

# But still recommended to define these
LuxCore.parameterlength(l::Linear) = l.out_dims * l.in_dims + l.out_dims

LuxCore.statelength(::Linear) = 0
```

```ansi
Parameter Length: 12; State Length: 0
```

::: tip No RNG in `parameterlength` and `statelength`

You might notice that we don't pass in a `RNG` for these functions. If your parameter length and/or state length depend on a random number generator, you should think **really hard** about what you are trying to do and why.

:::

Now, we need to define how the layer works. For this you make your layer a function with exactly 3 arguments – `x` the input, `ps` the parameters, and `st` the states. This function must return two things – `y` the output, and `st_new` the updated state.

```julia
function (l::Linear)(x::AbstractMatrix, ps, st::NamedTuple)
    y = ps.weight * x .+ ps.bias
    return y, st
end
```

Finally, let's run this layer. If you have made this far into the documentation, we don't feel you need a refresher on that. But if you do see the [Quickstart](/introduction/index#Quickstart).

```julia
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = LuxCore.setup(rng, l)

println("Parameter Length: ", LuxCore.parameterlength(l), "; State Length: ",
    LuxCore.statelength(l))

x = randn(rng, Float32, 2, 1)

LuxCore.apply(l, x, ps, st) # or `l(x, ps, st)`
```

```ansi
(Float32[-0.15276335; 0.45325348; 1.0207279; 0.78226817;;], NamedTuple())
```

### Container Layer {#Container-Layer}

If your layer comprises of other Lux layers, then it is a `Container Layer`. Note that you could treat it as a [`Singular Layer`](/manual/interface#singular-layer), and it is still fine. FWIW, if you cannot subtype your layer with `LuxCore.AbstractLuxContainerLayer` then you should go down the [`Singular Layer`](/manual/interface#singular-layer) route. But subtyping allows us to bypass some of these common definitions. Let us now define a layer, which is basically a composition of two linear layers.

::: tip Wrapper Layer

If you are defining a layer that is a wrapper around another layer, then you should subtype `LuxCore.AbstractLuxWrapperLayer` instead of `LuxCore.AbstractLuxContainerLayer`. The only difference from a container layer is that it can wrap a single layer and the parameter/state structure is exactly the same as the wrapped layer.

:::

```julia
struct ComposedLinear{L1, L2} <: LuxCore.AbstractLuxContainerLayer{(:linear_1, :linear_2)}
    linear_1::L1
    linear_2::L2
end

function (cl::ComposedLinear)(x::AbstractMatrix, ps, st::NamedTuple)
    # To access the parameters and states for `linear_1` we do `ps.linear_1` and
    # `st.linear_1`. Similarly for `linear_2`
    y, st_l1 = cl.linear_1(x, ps.linear_1, st.linear_1)
    y, st_l2 = cl.linear_2(y, ps.linear_2, st.linear_2)
    # Finally, we need to return the new state which has the exact structure as `st`
    return y, (linear_1 = st_l1, linear_2 = st_l2)
end
```

Here, you will notice we have passed `(:linear_1, :linear_2)` to the supertype. It essentially informs the type that, `<obj>.linear_1` and `<obj>.linear_2` are Lux layers and we need to construct parameters and states for those. Let's construct these and see:

```julia
model = ComposedLinear(Linear(2, 4), Linear(4, 2))
display(model)

ps, st = LuxCore.setup(rng, model)

println("Parameters: ", ps)
println("States: ", st)

println("Parameter Length: ", LuxCore.parameterlength(model), "; State Length: ",
    LuxCore.statelength(model))

x = randn(rng, Float32, 2, 1)

LuxCore.apply(model, x, ps, st) # or `model(x, ps, st)`
```

```ansi
(Float32[1.3410565; 0.78000563;;], (linear_1 = NamedTuple(), linear_2 = NamedTuple()))
```

## Parameter Interface {#Parameter-Interface}

We accept any parameter type as long as we can fetch the parameters using `getproperty(obj, :parameter_name)`. This allows us to simultaneously support `NamedTuple`s and `ComponentArray`s. Let us go through a concrete example of what it means. Consider [`Dense`](/api/Lux/layers#Lux.Dense) which expects two parameters named `weight` and `bias`.

::: tip Automatic Differentiation

If you are defining your own parameter type, it is your responsibility to make sure that it works with the AutoDiff System you are using.

:::

```julia
using Lux, Random

d = Dense(2, 3)
rng = Random.default_rng()
Random.seed!(rng, 0)

ps_default, st = LuxCore.setup(rng, d)

x = randn(rng, Float32, 2, 1)

println("Result with `NamedTuple` parameters: ", first(d(x, ps_default, st)))
```

```ansi
Result with `NamedTuple` parameters: Float32[-0.08713347; -0.4851346; -0.8490221;;]
```

Let, us define a custom parameter type with fields `myweight` and `mybias` but if we try to access `weight` we get back `myweight`, similar for `bias`.

::: warning Beware!

This is for demonstrative purposes, don't try this at home!

:::

```julia
struct DenseLayerParameters{W, B}
    myweight::W
    mybias::B
end

function Base.getproperty(ps::DenseLayerParameters, x::Symbol)
    if x == :weight
        return getfield(ps, :myweight)
    elseif x == :bias
        return getfield(ps, :mybias)
    end
    return getfield(ps, x)
end

ps = DenseLayerParameters(ps_default.weight, ps_default.bias)

println("Result with `DenseLayerParameters` parameters: ", first(d(x, ps, st)))
```

```ansi
Result with `DenseLayerParameters` parameters: Float32[0.23710957; 0.1003911; -0.57671577;;]
```

The takeaway from this shouldn't be – *lets define weird parameter types*. Simply because you can do weird things like this doesn't mean you should, since it only leads to bugs.

Instead this shows the flexibility you have for how your parameters can be structured.

## State Interface {#State-Interface}

States are always type constrained to be `NamedTuple`. The structure of the input state **must** match that of the output state, i.e. `keys(st_in) == keys(st_out)`. This doesn't imply that types of the input and output state match. To generate efficient code, we often do dispatch on the state, for example, [`Lux.Dropout`](/api/Lux/layers#Lux.Dropout), [`BatchNorm`](/api/Lux/layers#Lux.BatchNorm), etc.

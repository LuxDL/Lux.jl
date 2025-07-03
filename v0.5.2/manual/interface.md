
<a id='Lux Interface'></a>

# Lux Interface


:::tip


If you just want to define compatibility with Lux without actually using any of the other functionality provided by Lux (like layers), it is recommended to depend on `LuxCore.jl` instead of `Lux.jl`. `LuxCore.jl` is a significantly lighter dependency.


:::


First let's set the expectations straight.


  * Do you **have to** follow the interface? *No*.
  * **Should you** follow it? *Probably yes*.
  * **Why?** It provides the ability for frameworks built on top of Lux to be cross compatible. Additionally, any new functionality built into Lux, will just work for your framework.


::: warning


The interface is optional for frameworks being developed independent of Lux. All functionality in the core library (and officially supported ones) **must** adhere to the interface


:::


<a id='Layer Interface'></a>

## Layer Interface


<a id='Singular Layer'></a>

### Singular Layer


If the layer doesn't contain any other Lux layer, then it is a `Singular Layer`. This means it should optionally subtype `Lux.AbstractExplicitLayer` but mandatorily define all the necessary functions mentioned in the docstrings. Consider a simplified version of [`Dense`](../api/Lux/layers#Lux.Dense) called `Linear`.


First, setup the architectural details for this layer. Note, that the architecture doesn't contain any mutable structure like arrays. When in doubt, remember, once constructed a model architecture cannot change.


::: tip


For people coming from Flux.jl background this might be weird. We recommend checking out [the Flux to Lux migration guide](migrate_from_flux) first before proceeding.


:::


```julia
using Lux, Random

struct Linear{F1, F2} <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_weight::F1
    init_bias::F2
end

function Linear(in_dims::Int, out_dims::Int; init_weight=Lux.glorot_uniform,
    init_bias=Lux.zeros32)
    return Linear{typeof(init_weight), typeof(init_bias)}(in_dims, out_dims, init_weight,
        init_bias)
end

l = Linear(2, 4)
```


```
Linear()
```


Next, we need to implement functions which return the parameters and states for the layer. In case of `Linear`, the parameters are `weight` and `bias` while the states are empty. States become important when defining layers like [`BatchNorm`](../api/Lux/layers#Lux.BatchNorm), [`WeightNorm`](../api/Lux/layers#Lux.WeightNorm), etc. The recommended data structure for returning parameters is a NamedTuple, though anything satisfying the [Parameter Interface](#parameter-interface) is valid.


```julia
function Lux.initialparameters(rng::AbstractRNG, l::Linear)
    return (weight=l.init_weight(rng, l.out_dims, l.in_dims),
            bias=l.init_bias(rng, l.out_dims, 1))
end

Lux.initialstates(::AbstractRNG, ::Linear) = NamedTuple()
```


You could also implement `Lux.parameterlength` and `Lux.statelength` to prevent wasteful reconstruction of the parameters and states.


```julia
# This works
println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
    Lux.statelength(l))

# But still recommened to define these
Lux.parameterlength(l::Linear) = l.out_dims * l.in_dims + l.out_dims

Lux.statelength(::Linear) = 0
```


```
Parameter Length: 12; State Length: 0
```


::: tip


You might notice that we don't pass in a `PRNG` for these functions. If your parameter length and/or state length depend on a random number generator, you should think **really hard** about what you are trying to do and why.


:::


Now, we need to define how the layer works. For this you make your layer a function with exactly 3 arguments – `x` the input, `ps` the parameters, and `st` the states. This function must return two things – `y` the output, and `st_new` the updated state.


```julia
function (l::Linear)(x::AbstractMatrix, ps, st::NamedTuple)
    y = ps.weight * x .+ ps.bias
    return y, st
end
```


Finally, let's run this layer. If you have made this far into the documentation, we don't feel you need a refresher on that.


```julia
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, l)

println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
    Lux.statelength(l))

x = randn(rng, Float32, 2, 1)

Lux.apply(l, x, ps, st) # or `l(x, ps, st)`
```


```
(Float32[-0.15276335; 0.45325348; 1.0207279; 0.78226817;;], NamedTuple())
```


<a id='Container Layer'></a>

### Container Layer


If your layer comprises of other Lux layers, then it is a `Container Layer`. Note that you could treat it as a [`Singular Layer`](#singular-layer), and it is still fine. FWIW, if you cannot subtype your layer with `Lux.AbstractExplicitContainerLayer` then you should go down the [`Singular Layer`](#singular-layer) route. But subtyping allows us to bypass some of these common definitions. Let us now define a layer, which is basically a composition of two linear layers.


```julia
struct ComposedLinear{L1, L2} <: Lux.AbstractExplicitContainerLayer{(:linear_1, :linear_2)}
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

ps, st = Lux.setup(rng, model)

println("Parameters: ", ps)
println("States: ", st)

println("Parameter Length: ", Lux.parameterlength(model), "; State Length: ",
    Lux.statelength(model))

x = randn(rng, Float32, 2, 1)

Lux.apply(model, x, ps, st) # or `model(x, ps, st)`
```


```
(Float32[1.3410565; 0.78000563;;], (linear_1 = NamedTuple(), linear_2 = NamedTuple()))
```


<a id='Parameter Interface'></a>

## Parameter Interface


We accept any parameter type as long as we can fetch the parameters using `getproperty(obj, :parameter_name)`. This allows us to simultaneously support `NamedTuple`s and `ComponentArray`s. Let us go through a concrete example of what it means. Consider [`Dense`](../api/Lux/layers#Lux.Dense) which expects two parameters named `weight` and `bias`.


::: info


If you are defining your own parameter type, it is your responsibility to make sure that it works with the AutoDiff System you are using.


:::


```julia
using Lux, Random

d = Dense(2, 3)
rng = Random.default_rng()
Random.seed!(rng, 0)

ps_default, st = Lux.setup(rng, d)

x = randn(rng, Float32, 2, 1)

println("Result with `NamedTuple` parameters: ", first(d(x, ps_default, st)))
```


```
Result with `NamedTuple` parameters: Float32[1.135916; 0.7668784; -1.0876652;;]
```


Let, us define a custom parameter type with fields `myweight` and `mybias` but if we try to access `weight` we get back `myweight`, similar for `bias`.


::: warning


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


```
Result with `DenseLayerParameters` parameters: Float32[1.135916; 0.7668784; -1.0876652;;]
```


The takeaway from this shouldn't be – *lets define weird parameter types*. Simply because you can do weird things like this doesn't mean you should, since it only leads to bugs.


Instead this shows the flexibility you have for how your parameters can be structured.


<a id='State Interface'></a>

## State Interface


States are always type constrained to be `NamedTuple`. The structure of the input state **must** match that of the output state, i.e. `keys(st_in) == keys(st_out)`. This doesn't imply that types of the input and output state match. To generate efficient code, we often do dispatch on the state, for example, [`Dropout`](../api/Lux/layers#Lux.Dropout), [`BatchNorm`](../api/Lux/layers#Lux.BatchNorm), etc.


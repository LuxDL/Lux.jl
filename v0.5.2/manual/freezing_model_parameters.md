
<a id='Freezing Model Parameters'></a>

# Freezing Model Parameters


::: warning


API for freezing parameters should be considered experimental at this point.


:::


In this manual entry, we will go over how to freeze certain parameters in a model.


<a id='Freezing Layers of a Particular Kind'></a>

## Freezing Layers of a Particular Kind


To freeze a particular kind of layer, let's say [`Dense`](../api/Lux/layers#Lux.Dense) in the following example. We can use [`Lux.Experimental.@layer_map`](../api/Lux/contrib#Lux.Experimental.@layer_map) and freeze layers if they are of type `Dense`.


```julia
using Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Chain(Dense(3, 4), Chain(Dense(4, 4), Dropout(0.5f0), BatchNorm(4)),
    Dense(4, 1); disable_optimizations=true)

ps, st = Lux.setup(rng, model)

x = randn(rng, Float32, 3, 2)

model(x, ps, st)

function freeze_dense(d::Lux.Dense, ps, st, ::String)
    return Lux.freeze(d, ps, st, (:weight, :bias))
end
freeze_dense(l, ps, st, name) = (l, ps, st)

model_frozen, ps_frozen, st_frozen = Lux.Experimental.@layer_map freeze_dense model ps st

model_frozen(x, ps_frozen, st_frozen)
```


```
(Float32[1.7641534 -1.7641534], (layer_1 = (frozen_params = (weight = Float32[-0.026350189 -0.5554656 -0.35653266; -0.17461072 0.6705545 0.29924855; -0.8935247 -0.42453378 -0.3020351; -0.7988979 -0.7666331 -0.7104237], bias = Float32[0.0; 0.0; 0.0; 0.0;;]), states = NamedTuple()), layer_2 = (layer_1 = (frozen_params = (weight = Float32[-0.47289538 -0.680748 0.1764085 0.34383082; 0.42747158 -0.13819042 -0.109261915 -0.6143286; -0.35790488 -0.20881107 0.70390546 0.48137343; 0.82561636 0.38187847 0.05779423 -0.35181466], bias = Float32[0.0; 0.0; 0.0; 0.0;;]), states = NamedTuple()), layer_2 = (rng = Random.Xoshiro(0x87711e5ce1a49ffe, 0xa210b60ecab6b8c5, 0x436c749552fc8172, 0x03e9c7d813a9f096), training = Val{true}()), layer_3 = (running_mean = Float32[-0.04517859, 0.03484953, -0.004917746, 0.0074841487], running_var = Float32[0.94082206, 0.92428976, 0.90048367, 0.90112025], training = Val{true}())), layer_3 = (frozen_params = (weight = Float32[0.3981135 0.45468387 -0.07694905 0.8353388], bias = Float32[0.0;;]), states = NamedTuple())))
```


<a id='Freezing by Layer Name'></a>

## Freezing by Layer Name


When the function in `layer_map` is called, the 4th argument is the name of the layer. For example, if you want to freeze the 1st layer inside the inner Chain. The name for this would be `<model>.layer_2.layer_1`.


:::code-group


```julia [Freezing by Layer Name]

function freeze_by_name(d, ps, st, name::String)
    if name == "model.layer_2.layer_1"
        return Lux.Experimental.freeze(d, ps, st, (:weight, :bias))
    else
        return d, ps, st
    end
end

```


```julia [Freezing by Layer Type]

function freeze_dense(d::Dense, ps, st, ::String)
    return Lux.Experimental.freeze(d, ps, st, (:weight, :bias))
end
freeze_dense(l, ps, st, _) = (l, ps, st)

```


:::


<a id='Freezing Part of the Parameters'></a>

## Freezing Part of the Parameters


Instead of freezing all the parameters, we can simply specify `(:weight,)` to freeze only the `weight` parameter while training the `bias` parameter.


::: code-group


```julia [Freezing Some Parameters of a Layer]

function freeze_by_name(d, ps, st, name::String)
    if name == "model.layer_2.layer_1"
        return Lux.freeze(d, ps, st, (:weight,))
    else
        return d, ps, st
    end
end

```


```julia [Freezing All Parameters of a Layer]

function freeze_by_name(d, ps, st, name::String)
    if name == "model.layer_2.layer_1"
        return Lux.freeze(d, ps, st, (:weight, :bias))
    else
        return d, ps, st
    end
end

```


:::


<a id='Freezing Part of a Chain'></a>

## Freezing Part of a Chain


Starting `v0.4.22`, we can directly index into a `Chain`. So freezing a part of a `Chain`, is extremely easy.


```julia
using Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Chain(Dense(3, 4), Dense(4, 4), Dropout(0.5f0), BatchNorm(4), Dense(4, 1))

model_frozen = Chain(model[1:2], Lux.freeze(model[3:4]), model[5])
ps, st = Lux.setup(rng, model_frozen)

x = randn(rng, Float32, 3, 2)

model_frozen(x, ps, st)
```


```
(Float32[1.7641534 -1.7641534], (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = (frozen_params = (layer_3 = NamedTuple(), layer_4 = (scale = Float32[1.0, 1.0, 1.0, 1.0], bias = Float32[0.0, 0.0, 0.0, 0.0])), states = (layer_3 = (rng = Random.Xoshiro(0x87711e5ce1a49ffe, 0xa210b60ecab6b8c5, 0x436c749552fc8172, 0x03e9c7d813a9f096), training = Val{true}()), layer_4 = (running_mean = Float32[-0.04517859, 0.03484953, -0.004917746, 0.0074841487], running_var = Float32[0.94082206, 0.92428976, 0.90048367, 0.90112025], training = Val{true}()))), layer_4 = NamedTuple()))
```


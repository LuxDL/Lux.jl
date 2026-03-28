---
url: /dev/manual/freezing_model_parameters.md
---
# Freezing Model Parameters {#freezing-model-parameters}

::: warning Warning

API for freezing parameters should be considered experimental at this point.

:::

In this manual entry, we will go over how to freeze certain parameters in a model.

## Freezing Layers of a Particular Kind {#Freezing-Layers-of-a-Particular-Kind}

To freeze a particular kind of layer, let's say [`Dense`](/api/Lux/layers#Lux.Dense) in the following example. We can use [`Lux.Experimental.layer_map`](/api/Lux/contrib#Lux.Experimental.layer_map) and freeze layers if they are of type `Dense`.

```julia
using Lux, Random, Functors

rng = Xoshiro(0)

model = Chain(Dense(3, 4), Chain(Dense(4, 4), Dropout(0.5f0), BatchNorm(4)), Dense(4, 1))

ps, st = Lux.setup(rng, model)

x = randn(rng, Float32, 3, 2)

model(x, ps, st)

function freeze_dense(d::Lux.Dense, ps, st, path)
    return Lux.Experimental.freeze(d, ps, st, (:weight, :bias))
end
freeze_dense(l, ps, st, path) = (l, ps, st)

model_frozen, ps_frozen, st_frozen = Lux.Experimental.layer_map(freeze_dense, model, ps, st)

model_frozen(x, ps_frozen, st_frozen)
```

```ansi
(Float32[0.6886741 -1.2361472], (layer_1 = (frozen_params = (weight = Float32[-0.028461456 -0.5999714 -0.3850993; -0.18860114 0.72428167 0.32322538; -0.965117 -0.4585489 -0.32623518; -0.86290836 -0.82805836 -0.7673453], bias = Float32[0.4216236, -0.4510427, -0.097253, 0.23325463]), states = NamedTuple()), layer_2 = (layer_1 = (frozen_params = (weight = Float32[-0.680748 0.1764085 0.34383082 0.6469914; -0.13819042 -0.109261915 -0.6143286 -0.21672015; -0.20881107 0.70390546 0.48137343 0.25662464; 0.38187847 0.05779423 -0.35181466 -0.096988946], bias = Float32[0.41246277, 0.4318977, -0.4305781, 0.3367505]), states = NamedTuple()), layer_2 = (rng = Random.Xoshiro(0x4fa3403dd074e603, 0x12c522b8034ae186, 0x8e0c3a65079041bb, 0x21617f7747d97206, 0x22a21880af5dc689), training = Val{true}()), layer_3 = (running_mean = Float32[0.01965834, 0.0, 0.0, 0.015937408], running_var = Float32[0.90772897, 0.9, 0.9, 0.90508], training = Val{true}())), layer_3 = (frozen_params = (weight = Float32[0.7794657 0.8337032 0.6323408 -0.18308182], bias = Float32[-0.27373654]), states = NamedTuple())))
```

## Freezing by Layer Name {#Freezing-by-Layer-Name}

When the function in `layer_map` is called, the 4th argument is the name of the layer. For example, if you want to freeze the 1st layer inside the inner Chain. The name for this would be `layer_2.layer_1`.

:::code-group

```julia [Freezing by Layer Name]

function freeze_by_name(d, ps, st, name::KeyPath)
    name == KeyPath(:layer_2, :layer_1) &&
        return Lux.Experimental.freeze(d, ps, st, (:weight, :bias))
    return d, ps, st
end

```

```julia [Freezing by Layer Type]

function freeze_dense(d::Dense, ps, st, _)
    return Lux.Experimental.freeze(d, ps, st, (:weight, :bias))
end
freeze_dense(l, ps, st, _) = (l, ps, st)

```

:::

## Freezing Part of the Parameters {#Freezing-Part-of-the-Parameters}

Instead of freezing all the parameters, we can simply specify `(:weight,)` to freeze only the `weight` parameter while training the `bias` parameter.

::: code-group

```julia [Freezing Some Parameters of a Layer]

function freeze_by_name(d, ps, st, name::KeyPath)
    name == KeyPath(:layer_2, :layer_1) &&
        return Lux.Experimental.freeze(d, ps, st, (:weight,))
    return d, ps, st
end

```

```julia [Freezing All Parameters of a Layer]

function freeze_by_name(d, ps, st, name::KeyPath)
    name == KeyPath(:layer_2, :layer_1) &&
        return Lux.Experimental.freeze(d, ps, st, (:weight, :bias))
    return d, ps, st
end

```

:::

## Freezing Part of a Chain {#Freezing-Part-of-a-Chain}

```julia
using Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Chain(Dense(3, 4), Dense(4, 4), Dropout(0.5f0), BatchNorm(4), Dense(4, 1))

model_frozen = Chain(model[1:2], Lux.Experimental.freeze(model[3:4]), model[5])
ps, st = Lux.setup(rng, model_frozen)

x = randn(rng, Float32, 3, 2)

model_frozen(x, ps, st)
```

```ansi
(Float32[0.7429947 -1.2904677], (layer_1 = (layer_1 = NamedTuple(), layer_2 = NamedTuple()), layer_2 = (frozen_params = (layer_3 = NamedTuple(), layer_4 = (scale = Float32[1.0, 1.0, 1.0, 1.0], bias = Float32[0.0, 0.0, 0.0, 0.0])), states = (layer_3 = (rng = Random.TaskLocalRNG(), training = Val{true}()), layer_4 = (running_mean = Float32[0.0, 0.048522998, 0.0, 0.015937408], running_var = Float32[0.9, 0.9470896, 0.9, 0.90508], training = Val{true}()))), layer_3 = NamedTuple()))
```

# [Freezing Model Parameters](@id freezing-model-parameters)

!!! warning

    API for freezing parameters should be considered experimental at this point.

In this manual entry, we will go over how to freeze certain parameters in a model.

## Freezing Layers of a Particular Kind

To freeze a particular kind of layer, let's say [`Dense`](@ref) in the following example.
We can use [`Lux.Experimental.@layer_map`](@ref) and freeze layers if they are of type
`Dense`.

```@example
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

## Freezing by Layer Name

When the function in `layer_map` is called, the 4th argument is the name of the layer.
For example, if you want to freeze the 1st layer inside the inner Chain. The name for this
would be `<model>.layer_2.layer_1`.

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

## Freezing Part of the Parameters

Instead of freezing all the parameters, we can simply specify `(:weight,)` to freeze only
the `weight` parameter while training the `bias` parameter.

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

## Freezing Part of a Chain

Starting `v0.4.22`, we can directly index into a `Chain`. So freezing a part of a `Chain`,
is extremely easy.

```@example
using Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Chain(Dense(3, 4), Dense(4, 4), Dropout(0.5f0), BatchNorm(4), Dense(4, 1))

model_frozen = Chain(model[1:2], Lux.freeze(model[3:4]), model[5])
ps, st = Lux.setup(rng, model_frozen)

x = randn(rng, Float32, 3, 2)

model_frozen(x, ps, st)
```

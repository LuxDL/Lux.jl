# Freezing Model Parameters

!!! warning

    API for freezing parameters should be considered experimental at this point.

In this manual we will go over how to freeze certain parameters in a model.

## Freezing layers of a particular kind

To freeze a particular kind of layer, let's say [`Dense`](@ref) in the following example.
We can use the [`Lux.model_map`](@ref) function and freeze layers if they are of type
`Dense`.

```@example
import Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Lux.Chain(Lux.Dense(3, 4), Lux.Chain(Lux.Dense(4, 4), Lux.Dropout(0.5f0),
                                             Lux.BatchNorm(4)),
                  Lux.Dense(4, 1); disable_optimizations=true)

ps, st = Lux.setup(rng, model)

x = randn(rng, Float32, 3, 2)

model(x, ps, st)

function freeze_dense(d::Lux.Dense, ps, st, name::String)
    return Lux.freeze(d, ps, st, (:weight, :bias))
end
freeze_dense(l, ps, st, name) = (l, ps, st)

model_frozen, ps_frozen, st_frozen = Lux.model_map(freeze_dense, model, ps, st, "model")

model_frozen(x, ps_frozen, st_frozen)
```

## Freezing by layer name

When the function in `model_map` is called, the 4th argument is the name of the layer.
For example, if you want to freeze the 1st layer inside the inner Chain. The name for this
would be `<model>.layer_2.layer_1`.


```@raw html
=== "Freezing by layer name"

    ```julia
    function freeze_by_name(d, ps, st, name::String)
        if name == "model.layer_2.layer_1"
            return Lux.freeze(d, ps, st, (:weight, :bias))
        else
            return d, ps, st
        end
    end
    ```

=== "Freezing by layer type"

    ```julia
    function freeze_dense(d::Lux.Dense, ps, st, name::String)
        return Lux.freeze(d, ps, st, (:weight, :bias))
    end
    freeze_dense(l, ps, st, name) = (l, ps, st)
    ```
```

## Freezing part of the parameters

Instead of freezing all the parameters, we can simple specify `(:weight,)` to freeze only
the `weight` parameter while training the `bias` parameter.


```@raw html
=== "Freezing some parameters of a layer"

    ```julia hl_lines="3"
    function freeze_by_name(d, ps, st, name::String)
        if name == "model.layer_2.layer_1"
            return Lux.freeze(d, ps, st, (:weight,))
        else
            return d, ps, st
        end
    end
    ```

=== "Freezing all parameters of a layer"

    ```julia
    function freeze_by_name(d, ps, st, name::String)
        if name == "model.layer_2.layer_1"
            return Lux.freeze(d, ps, st, (:weight, :bias))
        else
            return d, ps, st
        end
    end
    ```
```
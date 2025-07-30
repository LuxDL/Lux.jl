# [Debugging Lux Models](@id debug-lux-layers)

Debugging DNNs can be very painful. Especially with the gigantic stacktraces for Lux, it is
even harder to pin-point to which particular layer errored out. This page describes some
useful tools that ship with Lux, that can help you debug your models.

!!! tip "TL;DR"

    Simply wrap your model with `Lux.Experimental.@debug_mode`!!

!!! warning "Don't Forget"

    Remember to use the non Debug mode model after you finish debugging. Debug mode models
    are way slower.

Let us construct a model which has an obviously incorrect dimension. In this example, you
will see how easy it is to pin-point the problematic layer.

## Incorrect Model Specification: Dimension Mismatch Problems

```@example manual_debugging
using Lux, Random

model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 3), Dense(1 => 1)), BatchNorm(1))

model_debug = Lux.Experimental.@debug_mode model
```

Note that we can use the parameters and states for `model` itself in `model_debug`, no need
to make any changes. If you ran the original model this is the kind of error you would see:

```@example manual_debugging
rng = Xoshiro(0)

ps, st = Lux.setup(rng, model)
x = randn(rng, Float32, 1, 2)

try
    model(x, ps, st)
catch e
    println(e)
end
```

Of course, this error will come with a detailed stacktrace, but it is still not very useful.
Now let's try using the debug mode model:

```@example manual_debugging
try
    model_debug(x, ps, st)
catch e
    println(e)
end
```

See now we know that `model.layers.layer_2.layers.layer_2` is the problematic layer. Let us
fix that layer and see what happens:

```julia
model = Chain(Dense(1 => 16, relu),
    Chain(
        Dense(16 => 3),  # [!code --]
        Dense(16 => 1),  # [!code ++]
        Dense(1 => 1)),
    BatchNorm(1))
```

```@example manual_debugging
model_fixed = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)),
    BatchNorm(1))

ps, st = Lux.setup(rng, model_fixed)

model_fixed(x, ps, st)
```

Voila!! We have tracked down and fixed the problem.

## Tracking down NaNs

Have you encountered those pesky little NaNs in your training? They are very hard to track
down. We will create an artificially simulate NaNs in our model and see how we can track
the offending layer.

We can set `nan_check` to `:forward`, `:backward` or `:both` to check for NaNs in the
debug model. (or even disable it by setting it to `:none`)

```@example manual_debugging
model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)),
    BatchNorm(1))

ps, st = Lux.setup(rng, model)

model_debug = Lux.Experimental.@debug_mode model nan_check=:both
```

Let us set a value in the parameter to `NaN`:

```@example manual_debugging
ps.layer_2.layer_2.weight[1, 1] = NaN
```

Now let us run the model

```@example manual_debugging
model(x, ps, st)
```

Ah as expected our output is `NaN`. But is is not very clear how to track where the first
`NaN` occurred. Let's run the debug model and check:

```@example manual_debugging
try
    model_debug(x, ps, st)
catch e
    println(e)
end
```

And we have figured it out! The first `NaN` occurred in the parameters of
`model.layers.layer_2.layers.layer_2`! But what if NaN occurs in the reverse pass! Let us
define a custom layer and introduce a fake NaN in the backward pass.

```@example manual_debugging
using ChainRulesCore, Zygote

const CRC = ChainRulesCore

offending_layer(x) = 2 .* x
```

```@example manual_debugging
model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), offending_layer), BatchNorm(1))

ps, st = Lux.setup(rng, model)

model(x, ps, st)
```

Let us define a custom backward pass to introduce some NaNs:

```@example manual_debugging
function CRC.rrule(::typeof(offending_layer), x)
    y = offending_layer(x)
    function ∇offending_layer(Δ)
        problematicΔ = CRC.@thunk begin
            Δ = CRC.unthunk(Δ)
            Δ[1] = NaN
            return Δ
        end
        return NoTangent(), problematicΔ
    end
    return y, ∇offending_layer
end
```

Let us compute the gradient of the layer now:

```@example manual_debugging
Zygote.gradient(ps -> sum(first(model(x, ps, st))), ps)
```

Oh no!! A `NaN` is present in the gradient of `ps`. Let us run the debug model and see
where the `NaN` occurred:

```@example manual_debugging
model_debug = Lux.Experimental.@debug_mode model nan_check=:both

try
    Zygote.gradient(ps -> sum(first(model_debug(x, ps, st))), ps)
catch e
    println(e)
end
```

And there you go our debug layer prints that the problem is in
`WrappedFunction(offending_layer) at location model.layers.layer_2.layers.layer_2`! Once
we fix the pullback of the layer, we will fix the NaNs.

## Conclusion

In this manual section, we have discussed tracking down errors in Lux models. We have
covered tracking incorrect model specifications and NaNs in forward and backward passes.
However, remember that this is an **Experimental** feature, and there might be edge cases
that don't work correctly. If you find any such cases, please open an issue on GitHub!

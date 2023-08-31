# Debugging Lux Models

Debugging DNNs can be very painful. Especially with the gigantic stacktraces for Lux, it is
even harder to pin-point to which particular layer errored out. This page describes some
useful tools that ship with Lux, that can help you debug your models.

:::tip TL;DR

Simply wrap your model with [`Lux.Experimental.@debug`](@ref)!!

:::

:::warning DON'T FORGET

Remember to use the non Debug mode model after you finish debugging. Debug mode models are
way slower.

:::

Let us construct a model which has an obviously incorrect dimension. In this example, you
will see how easy it is to pin-point the problematic layer.

```@example manual_debugging
using Lux, Random

model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 3), Dense(1 => 1)),
    BatchNorm(1); disable_optimizations=true)

model_debug = Lux.Experimental.@debug_mode model
```

Note that we can use the parameters and states for `model` itself in `model_debug`, no need
to make any changes. If you ran the original model this is the kind of error you would see:

```@example manual_debugging
rng = Xoshiro(0)

ps, st = Lux.setup(rng, model)
x = randn(rng, Float32, 1, 1)

try
    model(x, ps, st)
catch e
    println(e)
end
```

Ofcourse, this error will come with a detailed stacktrace, but it is still not very useful.
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
    Chain(Dense(16 => 3),  // [!code --]
    Chain(Dense(16 => 1),  // [!code ++]
        Dense(1 => 1)),
    BatchNorm(1); disable_optimizations=true)
```

```@example manual_debugging
model_fixed = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)),
    BatchNorm(1); disable_optimizations=true)

ps, st = Lux.setup(rng, model_fixed)

model_fixed(x, ps, st)
```

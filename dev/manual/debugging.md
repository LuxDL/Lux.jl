---
url: /dev/manual/debugging.md
---
# Debugging Lux Models {#debug-lux-layers}

Debugging DNNs can be very painful. Especially with the gigantic stacktraces for Lux, it is even harder to pin-point to which particular layer errored out. This page describes some useful tools that ship with Lux, that can help you debug your models.

::: tip TL;DR

Simply wrap your model with `Lux.Experimental.@debug_mode`!!

:::

::: warning Don't Forget

Remember to use the non Debug mode model after you finish debugging. Debug mode models are way slower.

:::

Let us construct a model which has an obviously incorrect dimension. In this example, you will see how easy it is to pin-point the problematic layer.

## Incorrect Model Specification: Dimension Mismatch Problems {#Incorrect-Model-Specification:-Dimension-Mismatch-Problems}

```julia
using Lux, Random

model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 3), Dense(1 => 1)), BatchNorm(1))

model_debug = Lux.Experimental.@debug_mode model
```

```ansi
Chain(
    layer_1 = DebugLayer(
        layer = Dense(1 => 16, relu),             [90m# 32 parameters[39m
    ),
    layer_2 = Chain(
        layer_1 = DebugLayer(
            layer = Dense(16 => 3),               [90m# 51 parameters[39m
        ),
        layer_2 = DebugLayer(
            layer = Dense(1 => 1),                [90m# 2 parameters[39m
        ),
    ),
    layer_3 = DebugLayer(
        layer = BatchNorm(1, affine=true, track_stats=true),  [90m# 2 parameters[39m[90m, plus 3 non-trainable[39m
    ),
) [90m        # Total: [39m87 parameters,
[90m          #        plus [39m3 states.
```

Note that we can use the parameters and states for `model` itself in `model_debug`, no need to make any changes. If you ran the original model this is the kind of error you would see:

```julia
rng = Xoshiro(0)

ps, st = Lux.setup(rng, model)
x = randn(rng, Float32, 1, 2)

try
    model(x, ps, st)
catch e
    println(e)
end
```

```ansi
DimensionMismatch("A has shape (1, 1) but B has shape (3, 2)")
```

Of course, this error will come with a detailed stacktrace, but it is still not very useful. Now let's try using the debug mode model:

```julia
try
    model_debug(x, ps, st)
catch e
    println(e)
end
```

```ansi
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (1, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: Dense(1 => 16, relu) at location KeyPath(:model, :layers, :layer_1)!
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mOutput Type: Matrix{Float32} | Output Structure: (16, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (16, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: Dense(16 => 3) at location KeyPath(:model, :layers, :layer_2, :layers, :layer_1)!
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mOutput Type: Matrix{Float32} | Output Structure: (3, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (3, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: Dense(1 => 1) at location KeyPath(:model, :layers, :layer_2, :layers, :layer_2)!
[91m[1m┌ [22m[39m[91m[1mError: [22m[39mLayer Dense(1 => 1) failed!! This layer is present at location KeyPath(:model, :layers, :layer_2, :layers, :layer_2).
[91m[1m└ [22m[39m[90m@ Lux.Experimental ~/work/Lux.jl/Lux.jl/src/contrib/debug.jl:116[39m
DimensionMismatch("A has shape (1, 1) but B has shape (3, 2)")
```

See now we know that `model.layers.layer_2.layers.layer_2` is the problematic layer. Let us fix that layer and see what happens:

```julia
model = Chain(Dense(1 => 16, relu),
    Chain(
        Dense(16 => 3),  # [!code --]
        Dense(16 => 1),  # [!code ++]
        Dense(1 => 1)),
    BatchNorm(1))
```

```julia
model_fixed = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)),
    BatchNorm(1))

ps, st = Lux.setup(rng, model_fixed)

model_fixed(x, ps, st)
```

```ansi
(Float32[-0.99998605 0.999986], (layer_1 = NamedTuple(), layer_2 = (layer_1 = NamedTuple(), layer_2 = NamedTuple()), layer_3 = (running_mean = Float32[0.07133968], running_var = Float32[0.971899], training = Val{true}())))
```

Voila!! We have tracked down and fixed the problem.

## Tracking down NaNs {#Tracking-down-NaNs}

Have you encountered those pesky little NaNs in your training? They are very hard to track down. We will create an artificially simulate NaNs in our model and see how we can track the offending layer.

We can set `nan_check` to `:forward`, `:backward` or `:both` to check for NaNs in the debug model. (or even disable it by setting it to `:none`)

```julia
model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)),
    BatchNorm(1))

ps, st = Lux.setup(rng, model)

model_debug = Lux.Experimental.@debug_mode model nan_check=:both
```

```ansi
Chain(
    layer_1 = DebugLayer(
        layer = Dense(1 => 16, relu),             [90m# 32 parameters[39m
    ),
    layer_2 = Chain(
        layer_1 = DebugLayer(
            layer = Dense(16 => 1),               [90m# 17 parameters[39m
        ),
        layer_2 = DebugLayer(
            layer = Dense(1 => 1),                [90m# 2 parameters[39m
        ),
    ),
    layer_3 = DebugLayer(
        layer = BatchNorm(1, affine=true, track_stats=true),  [90m# 2 parameters[39m[90m, plus 3 non-trainable[39m
    ),
) [90m        # Total: [39m53 parameters,
[90m          #        plus [39m3 states.
```

Let us set a value in the parameter to `NaN`:

```julia
ps.layer_2.layer_2.weight[1, 1] = NaN
```

```ansi
NaN
```

Now let us run the model

```julia
model(x, ps, st)
```

```ansi
(Float32[NaN NaN], (layer_1 = NamedTuple(), layer_2 = (layer_1 = NamedTuple(), layer_2 = NamedTuple()), layer_3 = (running_mean = Float32[NaN], running_var = Float32[NaN], training = Val{true}())))
```

Ah as expected our output is `NaN`. But is is not very clear how to track where the first `NaN` occurred. Let's run the debug model and check:

```julia
try
    model_debug(x, ps, st)
catch e
    println(e)
end
```

```ansi
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (1, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: Dense(1 => 16, relu) at location KeyPath(:model, :layers, :layer_1)!
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mOutput Type: Matrix{Float32} | Output Structure: (16, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (16, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: Dense(16 => 1) at location KeyPath(:model, :layers, :layer_2, :layers, :layer_1)!
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mOutput Type: Matrix{Float32} | Output Structure: (1, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (1, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: Dense(1 => 1) at location KeyPath(:model, :layers, :layer_2, :layers, :layer_2)!
DomainError(Float32[NaN;;], "NaNs detected in parameters (@ KeyPath(:weight,))  of layer Dense(1 => 1) at location KeyPath(:model, :layers, :layer_2, :layers, :layer_2).")
```

And we have figured it out! The first `NaN` occurred in the parameters of `model.layers.layer_2.layers.layer_2`! But what if NaN occurs in the reverse pass! Let us define a custom layer and introduce a fake NaN in the backward pass.

```julia
using ChainRulesCore, Zygote

const CRC = ChainRulesCore

offending_layer(x) = 2 .* x
```

```ansi
offending_layer (generic function with 1 method)
```

```julia
model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), offending_layer), BatchNorm(1))

ps, st = Lux.setup(rng, model)

model(x, ps, st)
```

```ansi
(Float32[0.9999881 -0.9999881], (layer_1 = NamedTuple(), layer_2 = (layer_1 = NamedTuple(), layer_2 = NamedTuple()), layer_3 = (running_mean = Float32[0.0026271285], running_var = Float32[0.98396176], training = Val{true}())))
```

Let us define a custom backward pass to introduce some NaNs:

```julia
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

```julia
Zygote.gradient(ps -> sum(first(model(x, ps, st))), ps)
```

```ansi
((layer_1 = (weight = Float32[0.0; 0.0; … ; 0.0; 0.0;;], bias = Float32[0.0, 0.0, 0.0, 0.0, NaN, 0.0, NaN, NaN, 0.0, 0.0, 0.0, NaN, NaN, NaN, 0.0, 0.0]), layer_2 = (layer_1 = (weight = Float32[NaN NaN … NaN NaN], bias = Float32[NaN]), layer_2 = nothing), layer_3 = (scale = Float32[0.0], bias = Float32[2.0])),)
```

Oh no!! A `NaN` is present in the gradient of `ps`. Let us run the debug model and see where the `NaN` occurred:

```julia
model_debug = Lux.Experimental.@debug_mode model nan_check=:both

try
    Zygote.gradient(ps -> sum(first(model_debug(x, ps, st))), ps)
catch e
    println(e)
end
```

```ansi
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (1, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: Dense(1 => 16, relu) at location KeyPath(:model, :layers, :layer_1)!
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mOutput Type: Matrix{Float32} | Output Structure: (16, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (16, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: Dense(16 => 1) at location KeyPath(:model, :layers, :layer_2, :layers, :layer_1)!
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mOutput Type: Matrix{Float32} | Output Structure: (1, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (1, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: WrappedFunction(offending_layer) at location KeyPath(:model, :layers, :layer_2, :layers, :layer_2)!
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mOutput Type: Matrix{Float32} | Output Structure: (1, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mInput Type: Matrix{Float32} | Input Structure: (1, 2).
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRunning Layer: BatchNorm(1, affine=true, track_stats=true) at location KeyPath(:model, :layers, :layer_3)!
[36m[1m[ [22m[39m[36m[1mInfo: [22m[39mOutput Type: Matrix{Float32} | Output Structure: (1, 2).
DomainError(Float32[NaN 0.0], "NaNs detected in pullback output (x)  of layer WrappedFunction(offending_layer) at location KeyPath(:model, :layers, :layer_2, :layers, :layer_2).")
```

And there you go our debug layer prints that the problem is in `WrappedFunction(offending_layer) at location model.layers.layer_2.layers.layer_2`! Once we fix the pullback of the layer, we will fix the NaNs.

## Conclusion {#Conclusion}

In this manual section, we have discussed tracking down errors in Lux models. We have covered tracking incorrect model specifications and NaNs in forward and backward passes. However, remember that this is an **Experimental** feature, and there might be edge cases that don't work correctly. If you find any such cases, please open an issue on GitHub!

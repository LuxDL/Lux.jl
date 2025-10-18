
<a id='Debugging-Lux-Models'></a>

# Debugging Lux Models


Debugging DNNs can be very painful. Especially with the gigantic stacktraces for Lux, it is even harder to pin-point to which particular layer errored out. This page describes some useful tools that ship with Lux, that can help you debug your models.


:::tip TL;DR


Simply wrap your model with `Lux.Experimental.@debug`!!


:::


:::warning DON'T FORGET


Remember to use the non Debug mode model after you finish debugging. Debug mode models are way slower.


:::


Let us construct a model which has an obviously incorrect dimension. In this example, you will see how easy it is to pin-point the problematic layer.


<a id='Incorrect-Model-Specification:-Dimension-Mismatch-Problems'></a>

## Incorrect Model Specification: Dimension Mismatch Problems


```julia
using Lux, Random

model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 3), Dense(1 => 1)),
    BatchNorm(1); disable_optimizations=true)

model_debug = Lux.Experimental.@debug_mode model
```


```
Chain(
    layer_1 = DebugLayer(
        layer = Dense(1 => 16, relu),   # 32 parameters
    ),
    layer_2 = Chain(
        layer_1 = DebugLayer(
            layer = Dense(16 => 3),     # 51 parameters
        ),
        layer_2 = DebugLayer(
            layer = Dense(1 => 1),      # 2 parameters
        ),
    ),
    layer_3 = DebugLayer(
        layer = BatchNorm(1, affine=true, track_stats=true),  # 2 parameters, plus 3
    ),
)         # Total: 87 parameters,
          #        plus 3 states.
```


Note that we can use the parameters and states for `model` itself in `model_debug`, no need to make any changes. If you ran the original model this is the kind of error you would see:


```julia
rng = Xoshiro(0)

ps, st = Lux.setup(rng, model)
x = randn(rng, Float32, 1, 1)

try
    model(x, ps, st)
catch e
    println(e)
end
```


```
DimensionMismatch("A has dimensions (1,1) but B has dimensions (3,1)")
```


Ofcourse, this error will come with a detailed stacktrace, but it is still not very useful. Now let's try using the debug mode model:


```julia
try
    model_debug(x, ps, st)
catch e
    println(e)
end
```


```
[ Info: Input Type: Matrix{Float32} | Input Structure: (1, 1)
[ Info: Running Layer: Dense(1 => 16, relu) at location model.layers.layer_1!
[ Info: Output Type: Matrix{Float32} | Output Structure: (16, 1)
[ Info: Input Type: Matrix{Float32} | Input Structure: (16, 1)
[ Info: Running Layer: Dense(16 => 3) at location model.layers.layer_2.layers.layer_1!
[ Info: Output Type: Matrix{Float32} | Output Structure: (3, 1)
[ Info: Input Type: Matrix{Float32} | Input Structure: (3, 1)
[ Info: Running Layer: Dense(1 => 1) at location model.layers.layer_2.layers.layer_2!
┌ Error: Layer Dense(1 => 1) failed!! This layer is present at location model.layers.layer_2.layers.layer_2
└ @ Lux.Experimental /var/lib/buildkite-agent/builds/gpuci-10/julialang/lux-dot-jl/src/contrib/debug.jl:113
DimensionMismatch("A has dimensions (1,1) but B has dimensions (3,1)")
```


See now we know that `model.layers.layer_2.layers.layer_2` is the problematic layer. Let us fix that layer and see what happens:


```julia
model = Chain(Dense(1 => 16, relu),
    Chain(Dense(16 => 3),  // [!code --]
    Chain(Dense(16 => 1),  // [!code ++]
        Dense(1 => 1)),
    BatchNorm(1); disable_optimizations=true)
```


```julia
model_fixed = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)),
    BatchNorm(1); disable_optimizations=true)

ps, st = Lux.setup(rng, model_fixed)

model_fixed(x, ps, st)
```


```
(Float32[0.0;;], (layer_1 = NamedTuple(), layer_2 = (layer_1 = NamedTuple(), layer_2 = NamedTuple()), layer_3 = (running_mean = Float32[-0.01397949], running_var = Float32[NaN], training = Val{true}())))
```


Voila!! We have tracked down and fixed the problem.


<a id='Tracking-down-NaNs'></a>

## Tracking down NaNs


Have you encountered those pesky little NaNs in your training? They are very hard to track down. We will create an artificially simulate NaNs in our model and see how we can track the offending layer.


We can set `nan_check` to `:forward`, `:backward` or `:both` to check for NaNs in the debug model. (or even disable it by setting it to `:none`)


```julia
model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)),
    BatchNorm(1); disable_optimizations=true)

ps, st = Lux.setup(rng, model)

model_debug = Lux.Experimental.@debug_mode model nan_check=:both
```


```
Chain(
    layer_1 = DebugLayer(
        layer = Dense(1 => 16, relu),   # 32 parameters
    ),
    layer_2 = Chain(
        layer_1 = DebugLayer(
            layer = Dense(16 => 1),     # 17 parameters
        ),
        layer_2 = DebugLayer(
            layer = Dense(1 => 1),      # 2 parameters
        ),
    ),
    layer_3 = DebugLayer(
        layer = BatchNorm(1, affine=true, track_stats=true),  # 2 parameters, plus 3
    ),
)         # Total: 53 parameters,
          #        plus 3 states.
```


Let us set a value in the parameter to `NaN`:


```julia
ps.layer_2.layer_2.weight[1, 1] = NaN
```


```
NaN
```


Now let us run the model


```julia
model(x, ps, st)
```


```
(Float32[NaN;;], (layer_1 = NamedTuple(), layer_2 = (layer_1 = NamedTuple(), layer_2 = NamedTuple()), layer_3 = (running_mean = Float32[NaN], running_var = Float32[NaN], training = Val{true}())))
```


Ah as expected our output is `NaN`. But is is not very clear how to track where the first `NaN` occurred. Let's run the debug model and check:


```julia
try
    model_debug(x, ps, st)
catch e
    println(e)
end
```


```
[ Info: Input Type: Matrix{Float32} | Input Structure: (1, 1)
[ Info: Running Layer: Dense(1 => 16, relu) at location model.layers.layer_1!
[ Info: Output Type: Matrix{Float32} | Output Structure: (16, 1)
[ Info: Input Type: Matrix{Float32} | Input Structure: (16, 1)
[ Info: Running Layer: Dense(16 => 1) at location model.layers.layer_2.layers.layer_1!
[ Info: Output Type: Matrix{Float32} | Output Structure: (1, 1)
[ Info: Input Type: Matrix{Float32} | Input Structure: (1, 1)
[ Info: Running Layer: Dense(1 => 1) at location model.layers.layer_2.layers.layer_2!
DomainError((weight = Float32[NaN;;], bias = Float32[0.0;;]), "NaNs detected in parameters of layer Dense(1 => 1) at location model.layers.layer_2.layers.layer_2")
```


And we have figured it out! The first `NaN` occurred in the parameters of `model.layers.layer_2.layers.layer_2`! But what if NaN occurs in the reverse pass! Let us define a custom layer and introduce a fake NaN in the backward pass.


```julia
using ChainRulesCore, Zygote

const CRC = ChainRulesCore

offending_layer(x) = 2 .* x
```


```
offending_layer (generic function with 1 method)
```


```julia
model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), offending_layer),
    BatchNorm(1); disable_optimizations=true)

ps, st = Lux.setup(rng, model)

model(x, ps, st)
```


```
(Float32[0.0;;], (layer_1 = NamedTuple(), layer_2 = (layer_1 = NamedTuple(), layer_2 = NamedTuple()), layer_3 = (running_mean = Float32[-0.092828535], running_var = Float32[NaN], training = Val{true}())))
```


Let us define a custom backward pass to introduce some NaNs:


```julia
function CRC.rrule(::typeof(offending_layer), x)
    y = offending_layer(x)
    function ∇offending_layer(Δ)
        Δ[1] = NaN
        return NoTangent(), Δ
    end
    return y, ∇offending_layer
end
```


Let us compute the gradient of the layer now:


```julia
Zygote.gradient(ps -> sum(first(model(x, ps, st))), ps)
```


```
((layer_1 = (weight = Float32[0.0; NaN; … ; NaN; 0.0;;], bias = Float32[0.0; NaN; … ; NaN; 0.0;;]), layer_2 = (layer_1 = (weight = Float32[NaN NaN … NaN NaN], bias = Float32[NaN;;]), layer_2 = nothing), layer_3 = (scale = Float32[0.0], bias = Fill(1.0f0, 1))),)
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


```
[ Info: Input Type: Matrix{Float32} | Input Structure: (1, 1)
[ Info: Running Layer: Dense(1 => 16, relu) at location model.layers.layer_1!
[ Info: Output Type: Matrix{Float32} | Output Structure: (16, 1)
[ Info: Input Type: Matrix{Float32} | Input Structure: (16, 1)
[ Info: Running Layer: Dense(16 => 1) at location model.layers.layer_2.layers.layer_1!
[ Info: Output Type: Matrix{Float32} | Output Structure: (1, 1)
[ Info: Input Type: Matrix{Float32} | Input Structure: (1, 1)
[ Info: Running Layer: WrappedFunction(offending_layer) at location model.layers.layer_2.layers.layer_2!
[ Info: Output Type: Matrix{Float32} | Output Structure: (1, 1)
[ Info: Input Type: Matrix{Float32} | Input Structure: (1, 1)
[ Info: Running Layer: BatchNorm(1, affine=true, track_stats=true) at location model.layers.layer_3!
[ Info: Output Type: Matrix{Float32} | Output Structure: (1, 1)
DomainError(Float32[NaN;;], "NaNs detected in pullback output for WrappedFunction(offending_layer) at location model.layers.layer_2.layers.layer_2!")
```


And there you go our debug layer prints that the problem is in `WrappedFunction(offending_layer) at location model.layers.layer_2.layers.layer_2`! Once we fix the pullback of the layer, we will fix the NaNs.


<a id='Conclusion'></a>

## Conclusion


In this manual section, we have discussed tracking down errors in Lux models. We have covered tracking incorrect model specifications and NaNs in forward and backward passes. However, remember that this is an **Experimental** feature, and there might be edge cases that don't work correctly. If you find any such cases, please open an issue on GitHub!


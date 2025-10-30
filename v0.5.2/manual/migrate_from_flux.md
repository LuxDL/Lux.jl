
<a id='Migrating from Flux to Lux'></a>

# Migrating from Flux to Lux


For the core library layers like [`Dense`](../api/Lux/layers#Lux.Dense), [`Conv`](../api/Lux/layers#Lux.Conv), etc. we have intentionally kept the API very similar to Flux. In most cases, replacing `using Flux` with `using Lux` should be enough to get you started. We cover the additional changes that you will have to make in the following example.


:::code-group


```julia{1,7,9,11} [Lux]
using Lux, Random, NNlib, Zygote

model = Chain(Dense(2 => 4), BatchNorm(4, relu), Dense(4 => 2))
rng = Random.default_rng()
x = randn(rng, Float32, 2, 4)

ps, st = Lux.setup(rng, model)

model(x, ps, st)

gradient(ps -> sum(first(model(x, ps, st))), ps)
```


```julia [Flux]
using Flux, Random, NNlib, Zygote

model = Chain(Dense(2 => 4), BatchNorm(4, relu), Dense(4 => 2))
rng = Random.default_rng()
x = randn(rng, Float32, 2, 4)



model(x)

gradient(model -> sum(model(x)), model)
```


:::


<a id='Implementing Custom Layers'></a>

## Implementing Custom Layers


Flux and Lux operate under extremely different design philosophies regarding how layers should be implemented. A summary of the differences would be:


  * Flux stores everything in a single struct and relies on `Functors.@functor` and `Flux.trainable` to distinguish between trainable and non-trainable parameters.
  * Lux relies on the user to define `Lux.initialparameters` and `Lux.initialstates` to distinguish between trainable parameters (called "parameters") and non-trainable parameters (called "states"). Additionally, Lux layers define the model architecture, hence device transfer utilities like [`gpu_device`](../api/LuxDeviceUtils/index#LuxDeviceUtils.gpu_device), [`cpu_device`](../api/LuxDeviceUtils/index#LuxDeviceUtils.cpu_device), etc. cannot be applied on Lux layers, instead they need to be applied on the parameters and states.


Let's work through a concrete example to demonstrate this. We will implement a very simple layer that computes $A \times B \times x$ where $A$ is not trainable and $B$ is trainable.


:::code-group


```julia [Lux]
using Lux, Random, NNlib, Zygote

struct LuxLinear <: Lux.AbstractExplicitLayer
    init_A
    init_B
end

function LuxLinear(A::AbstractArray, B::AbstractArray)
    # Storing Arrays or any mutable structure inside a Lux Layer is not recommended
    # instead we will convert this to a function to perform lazy initialization
    return LuxLinear(() -> copy(A), () -> copy(B))
end

# `B` is a parameter
Lux.initialparameters(::AbstractRNG, layer::LuxLinear) = (B=layer.init_B(),)

# `A` is a state
Lux.initialstates(::AbstractRNG, layer::LuxLinear) = (A=layer.init_A(),)

(l::LuxLinear)(x, ps, st) = st.A * ps.B * x, st
```


```julia [Flux]
using Flux, Random, NNlib, Zygote, Optimisers

struct FluxLinear
    A
    B
end







# `A` is not trainable
Optimisers.trainable(f::FluxLinear) = (B=f.B,)

# Needed so that both `A` and `B` can be transfered between devices
Flux.@functor FluxLinear

(l::FluxLinear)(x) = l.A * l.B * x
```


:::


Now let us run the model.


:::code-group


```julia{2,5,7,9} [Lux]
rng = Random.default_rng()
model = LuxLinear(randn(rng, 2, 4), randn(rng, 4, 2))
x = randn(rng, 2, 1)

ps, st = Lux.setup(rng, model)

model(x, ps, st)

gradient(ps -> sum(first(model(x, ps, st))), ps)
```


```julia [Flux]
rng = Random.default_rng()
model = FluxLinear(randn(rng, 2, 4), randn(rng, 4, 2))
x = randn(rng, 2, 1)



model(x)

gradient(model -> sum(model(x)), model)
```


:::


To reiterate some important points:


  * Don't store mutables like Arrays inside a Lux Layer.
  * Parameters and States should be constructured inside the respective `initial*` functions.


<a id='Certain Important Implementation Details'></a>

## Certain Important Implementation Details


<a id='Training/Inference Mode'></a>

### Training/Inference Mode


Flux supports a mode called `:auto` which automatically decides if the user is training the model or running inference. This is the default mode for `Flux.BatchNorm`, `Flux.GroupNorm`, `Flux.Dropout`, etc. Lux doesn't support this mode (specifically to keep code simple and do exactly what the user wants), hence our default mode is `training`. This can be changed using `Lux.testmode`.


<a id='Can we still use Flux Layers?'></a>

## Can we still use Flux Layers?


If you have `Flux` loaded in your code, you can use the function [`Lux.transform`](../api/Lux/flux_to_lux#Lux.transform) to automatically convert your model to `Lux`. Note that in case a native Lux counterpart isn't available, we fallback to using `Optimisers.destructure`.


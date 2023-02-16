# Migrating from Flux to Lux

For the core library layers like [`Dense`](@ref), [`Conv`](@ref), etc. we have intentionlly
kept the API very similar to Flux. In most cases, replacing `using Flux` with `using Lux`
should be enough to get you started. We cover the additional changes that you will have to
make in the following example.


```@raw html
=== "Lux"

    ```julia hl_lines="1 7 9 11"
    using Lux, Random, NNlib, Zygote

    model = Chain(Dense(2 => 4), BatchNorm(4, relu), Dense(4 => 2))
    rng = Random.default_rng()
    x = randn(rng, Float32, 2, 4)
  
    ps, st = Lux.setup(rng, model)

    model(x, ps, st)

    gradient(ps -> sum(first(model(x, ps, st))), ps)
    ```

=== "Flux"

    ```julia
    using Flux, Random, NNlib, Zygote

    model = Chain(Dense(2 => 4), BatchNorm(4, relu), Dense(4 => 2))
    rng = Random.default_rng()
    x = randn(rng, Float32, 2, 4)



    model(x)

    gradient(model -> sum(model(x)), model)
    ```
```

## Implementing Custom Layers

Flux and Lux operate under extremely different design philosophies regarding how layers
should be implemented. A summary of the differences would be:

* Flux stores everything in a single struct and relies on `Functors.@functor` and
  `Flux.trainable` to distinguish between trainable and non-trainable parameters.

* Lux relies on the user to define [`Lux.initialparameters`](@ref) and
  [`Lux.initialstates`](@ref) to distinguish between trainable parameters (called
  "parameters") and non-trainable parameters (called "states"). Additionally Lux layers
  define the model architecture, hence device transfer utilities like [`gpu`](@ref),
  [`cpu`](@ref), etc. cannot be applied on Lux layers, instead they need to be applied on
  the parameters and states.

Let's work through a concrete example to demonstrate this. We will implement a very simple
layer that computes ``A \times B \times x`` where ``A`` is not trainable and ``B`` is
trainable.

```@raw html
=== "Lux"

    ```julia
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
    Lux.initialparameters(rng::AbstractRNG, layer::LuxLinear) = (B=layer.init_B(),)

    # `A` is a state
    Lux.initialstates(rng::AbstractRNG, layer::LuxLinear) = (A=layer.init_A(),)

    (l::LuxLinear)(x, ps, st) = st.A * ps.B * x, st
    ```

=== "Flux"

    ```julia
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
```

Now let us run the model.

```@raw html
=== "Lux"

    ```julia hl_lines="2 5 7 9"
    rng = Random.default_rng()
    model = LuxLinear(randn(rng, 2, 4), randn(rng, 4, 2))
    x = randn(rng, 2, 1)

    ps, st = Lux.setup(rng, model)

    model(x, ps, st)

    gradient(ps -> sum(first(model(x, ps, st))), ps)
    ```

=== "Flux"

    ```julia
    rng = Random.default_rng()
    model = FluxLinear(randn(rng, 2, 4), randn(rng, 4, 2))
    x = randn(rng, 2, 1)



    model(x)

    gradient(model -> sum(model(x)), model)
    ```
```

To reiterate some of the important points:

* Don't store mutables like Arrays inside a Lux Layer.
* Parameters and States should be constructured inside the respective `initial*` functions.

## Certain Important Implementation Details

### Training/Inference Mode

Flux supports a mode called `:auto` which automatically decides if the user is training the
model or running inference. This is the default mode for `Flux.BatchNorm`, `Flux.GroupNorm`,
`Flux.Dropout`, etc. Lux doesn't support this mode (specifically to keep code simple and
do exactly what the user wants), hence our default mode is `training`. This can be changed
using [`Lux.testmode`](@ref).

## Can't access functions like `relu`, `sigmoid`, etc?

Unlike Flux we don't reexport functionality from `NNlib`, all you need to do to fix this is
add `using NNlib`.

## Missing some common layers from Flux

Lux is a very new framework, as such we haven't implemented all Layers that are a part of
Flux. We are tracking the missing features in
[this issue](https://github.com/avik-pal/Lux.jl/issues/13), and hope to have them
implemented soon. If you **really** need those functionality check out the next section.

## Can we still use Flux Layers?

We don't recommend this method, but here is a way to compose Flux with Lux.

!!! tip

    Starting `v0.4.37`, if you have `using Flux` in your code, Lux will automatically
    provide a function `transform` that can convert Flux layers to Lux layers

```julia
using Lux, NNlib, Random, Optimisers
import Flux

# Layer Implementation
struct FluxCompatLayer{L,I} <: Lux.AbstractExplicitLayer
    layer::L
    init_parameters::I
end

function FluxCompatLayer(flayer)
    p, re = Optimisers.destructure(flayer)
    p_ = copy(p)
    return FluxCompatLayer(re, () -> p_)
end

Lux.initialparameters(rng::AbstractRNG, l::FluxCompatLayer) = (p=l.init_parameters(),)

(f::FluxCompatLayer)(x, ps, st) = f.layer(ps.p)(x), st

# Running the model
fmodel = Flux.Chain(Flux.Dense(3 => 4, relu), Flux.Dense(4 => 1))

lmodel = FluxCompatLayer(fmodel)

rng = Random.default_rng()
x = randn(rng, 3, 1)

ps, st = Lux.setup(rng, lmodel)

lmodel(x, ps, st)[1] == fmodel(x)
```

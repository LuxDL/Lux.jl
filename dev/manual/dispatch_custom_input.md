---
url: /dev/manual/dispatch_custom_input.md
---
# Dispatching on Custom Input Types {#Dispatching-on-Custom-Input-Types}

## Which function should participate in dispatch? {#Which-function-should-participate-in-dispatch?}

* Defining a dispatch on `(::Layer)(x::MyInputType, ps, st::NamedTuple)` is inconvenient, since it requires the user to define a new method for every layer type.

* `(::AbstractLuxLayer)(x::MyInputType, ps, st::NamedTuple)` doesn't work.

* Instead, we need to define the dispatch on `Lux.apply(::AbstractLuxLayer, x::MyInputType, ps, st::NamedTuple)`.

## Concrete Example {#Concrete-Example}

Consider [Neural ODEs](https://implicit-layers-tutorial.org/neural_odes/). In these models, often time we want to every iteration of the neural network to take the current time as input. Here, we won't go through implementing an entire Neural ODE model. Instead we will define a time dependent version of [`Chain`](/api/Lux/layers#Lux.Chain).

### Time-Dependent Chain Implementation {#Time-Dependent-Chain-Implementation}

```julia
using Lux, Random

struct TDChain{L <: NamedTuple} <: Lux.AbstractLuxWrapperLayer{:layers}
    layers::L
end

function (l::TDChain)((x, t)::Tuple, ps, st::NamedTuple)
    # Concatenate along the 2nd last dimension
    sz = ntuple(i -> i == ndims(x) - 1 ? 1 : size(x, i), ndims(x))
    t_ = ones(eltype(x), sz) .* t  # Needs to be modified for GPU
    for name in keys(l.layers)
        x, st_ = Lux.apply(getfield(l.layers, name), cat(x, t_; dims=ndims(x) - 1),
                           getfield(ps, name), getfield(st, name))
        st = merge(st, NamedTuple{(name,)}((st_,)))
    end
    return x, st
end

model = Chain(Dense(3, 4), TDChain((; d1=Dense(5, 4), d2=Dense(5, 4))), Dense(4, 1))
```

```ansi
Chain(
    layer_1 = Dense(3 => 4),                      [90m# 16 parameters[39m
    layer_2 = TDChain(
        d(1-2) = Dense(5 => 4),                   [90m# 48 (24 x 2) parameters[39m
    ),
    layer_3 = Dense(4 => 1),                      [90m# 5 parameters[39m
) [90m        # Total: [39m69 parameters,
[90m          #        plus [39m0 states.
```

### Running the TDChain {#Running-the-TDChain}

```julia
rng = MersenneTwister(0)
ps, st = Lux.setup(rng, model)
x = randn(rng, Float32, 3, 2)

try
    model(x, ps, st)
catch e
    Base.showerror(stdout, e)
end
```

```ansi
MethodError: no method matching apply(::@NamedTuple{d1::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}, d2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, ::Matrix{Float32}, ::@NamedTuple{d1::@NamedTuple{weight::Matrix{Float32}, bias::Vector{Float32}}, d2::@NamedTuple{weight::Matrix{Float32}, bias::Vector{Float32}}}, ::@NamedTuple{d1::@NamedTuple{}, d2::@NamedTuple{}})
The function `apply` exists, but no method is defined for this combination of argument types.

[0mClosest candidates are:
[0m  apply([91m::AbstractLuxLayer[39m, ::Any, ::Any, ::Any)
[0m[90m   @[39m [35mLuxCore[39m [90m~/work/Lux.jl/Lux.jl/lib/LuxCore/src/[39m[90m[4mLuxCore.jl:154[24m[39m
[0m  apply([91m::StatefulLuxLayer[39m, ::Any, ::Any)
[0m[90m   @[39m [35mLuxCore[39m [90m~/work/Lux.jl/Lux.jl/lib/LuxCore/src/[39m[90m[4mstateful.jl:163[24m[39m
[0m  apply([91m::StatefulLuxLayer[39m, ::Any)
[0m[90m   @[39m [35mLuxCore[39m [90m~/work/Lux.jl/Lux.jl/lib/LuxCore/src/[39m[90m[4mstateful.jl:163[24m[39m
```

### Writing the Correct Dispatch Rules {#Writing-the-Correct-Dispatch-Rules}

* Create a Custom Layer storing the time.

  ```julia
  struct ArrayAndTime{A <: AbstractArray, T <: Real}
      array::A
      time::T
  end
  ```

* Define the dispatch on `Lux.apply(::AbstractLuxLayer, x::ArrayAndTime, ps, st::NamedTuple)`.

  ```julia
  function Lux.apply(layer::Lux.AbstractLuxLayer, x::ArrayAndTime, ps, st::NamedTuple)
      y, st = layer(x.array, ps, st)
      return ArrayAndTime(y, x.time), st
  end

  function Lux.apply(layer::TDChain, x::ArrayAndTime, ps, st::NamedTuple)
      y, st = layer((x.array, x.time), ps, st)
      return ArrayAndTime(y, x.time), st
  end
  ```

* Run the model.

  ```julia
  xt = ArrayAndTime(x, 10.0f0)

  model(xt, ps, st)[1]
  ```

  ```ansi
  Main.ArrayAndTime{Matrix{Float32}, Float32}(Float32[4.887438 5.5271416], 10.0f0)
  ```

### Using the Same Input for Non-TD Models {#Using-the-Same-Input-for-Non-TD-Models}

Writing proper dispatch means we can simply replace the `TDChain` with a `Chain` (of course with dimension corrections) and the pipeline still works.

```julia
model = Chain(Dense(3, 4), Chain((; d1=Dense(4, 4), d2=Dense(4, 4))), Dense(4, 1))

ps, st = Lux.setup(rng, model)

model(xt, ps, st)[1]
```

```ansi
Main.ArrayAndTime{Matrix{Float32}, Float32}(Float32[0.40721768 1.2363781], 10.0f0)
```

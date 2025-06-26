# Dispatching on Custom Input Types

## Which function should participate in dispatch?

* Defining a dispatch on `(::Layer)(x::MyInputType, ps, st::NamedTuple)` is inconvenient,
  since it requires the user to define a new method for every layer type.

* `(::AbstractLuxLayer)(x::MyInputType, ps, st::NamedTuple)` doesn't work.

* Instead, we need to define the dispatch on
  `Lux.apply(::AbstractLuxLayer, x::MyInputType, ps, st::NamedTuple)`.

## Concrete Example

Consider [Neural ODEs](https://implicit-layers-tutorial.org/neural_odes/). In these models,
often time we want to every iteration of the neural network to take the current time as
input. Here, we won't go through implementing an entire Neural ODE model. Instead we will
define a time dependent version of [`Chain`](@ref).

### Time-Dependent Chain Implementation

```@example dispatch
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

### Running the TDChain

```@example dispatch
rng = MersenneTwister(0)
ps, st = Lux.setup(rng, model)
x = randn(rng, Float32, 3, 2)

try
    model(x, ps, st)
catch e
    Base.showerror(stdout, e)
end
```

### Writing the Correct Dispatch Rules

* Create a Custom Layer storing the time.

  ```@example dispatch
  struct ArrayAndTime{A <: AbstractArray, T <: Real}
      array::A
      time::T
  end
  ```

* Define the dispatch on `Lux.apply(::AbstractLuxLayer, x::ArrayAndTime, ps, st::NamedTuple)`.

  ```@example dispatch
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

  ```@example dispatch
  xt = ArrayAndTime(x, 10.0f0)

  model(xt, ps, st)[1]
  ```

### Using the Same Input for Non-TD Models

Writing proper dispatch means we can simply replace the `TDChain` with a `Chain` (of course
with dimension corrections) and the pipeline still works.

```@example dispatch
model = Chain(Dense(3, 4), Chain((; d1=Dense(4, 4), d2=Dense(4, 4))), Dense(4, 1))

ps, st = Lux.setup(rng, model)

model(xt, ps, st)[1]
```

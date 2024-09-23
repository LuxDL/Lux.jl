
<a id='Dispatching on Custom Input Types'></a>

# Dispatching on Custom Input Types


<a id='Which function should participate in dispatch?'></a>

## Which function should participate in dispatch?


  * Defining a dispatch on `(::Layer)(x::MyInputType, ps, st::NamedTuple)` is inconvenient, since it requires the user to define a new method for every layer type.
  * `(::AbstractExplicitLayer)(x::MyInputType, ps, st::NamedTuple)` doesn't work.
  * Instead, we need to define the dispatch on `Lux.apply(::AbstractExplicitLayer, x::MyInputType, ps, st::NamedTuple)`.


<a id='Concrete Example'></a>

## Concrete Example


Consider [Neural ODEs](https://implicit-layers-tutorial.org/neural_odes/). In these models, often time we want to every iteration of the neural network to take the current time as input. Here, we won't go through implementing an entire Neural ODE model. Instead we will define a time dependent version of [`Chain`](../api/Lux/layers#Lux.Chain).


<a id='Time-Dependent Chain Implementation'></a>

### Time-Dependent Chain Implementation


```julia
using Lux, Random

struct TDChain{L <: NamedTuple} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
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


```
Chain(
    layer_1 = Dense(3 => 4),            # 16 parameters
    layer_2 = TDChain(
        layers = NamedTuple(
            d1 = Dense(5 => 4),         # 24 parameters
            d2 = Dense(5 => 4),         # 24 parameters
        ),
    ),
    layer_3 = Dense(4 => 1),            # 5 parameters
)         # Total: 69 parameters,
          #        plus 0 states.
```


<a id='Running the TDChain'></a>

### Running the TDChain


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


```
MethodError: no method matching (::Main.TDChain{NamedTuple{(:d1, :d2), Tuple{Dense{true, typeof(identity), typeof(glorot_uniform), typeof(zeros32)}, Dense{true, typeof(identity), typeof(glorot_uniform), typeof(zeros32)}}}})(::Matrix{Float32}, ::NamedTuple{(:d1, :d2), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, ::NamedTuple{(:d1, :d2), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}})

Closest candidates are:
  (::Main.TDChain)(!Matched::Tuple, ::Any, ::NamedTuple)
   @ Main dispatch_custom_input.md:29
```


<a id='Writing the Correct Dispatch Rules'></a>

### Writing the Correct Dispatch Rules


  * Create a Custom Layer storing the time.


```julia
struct ArrayAndTime{A <: AbstractArray, T <: Real}
    array::A
    time::T
end
```


  * Define the dispatch on `Lux.apply(::AbstractExplicitLayer, x::ArrayAndTime, ps, st::NamedTuple)`.


```julia
function Lux.apply(layer::Lux.AbstractExplicitLayer, x::ArrayAndTime, ps, st::NamedTuple)
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


```
Main.ArrayAndTime{Matrix{Float32}, Float32}(Float32[4.8016562 5.174927], 10.0f0)
```


<a id='Using the Same Input for Non-TD Models'></a>

### Using the Same Input for Non-TD Models


Writing proper dispatch means we can simply replace the `TDChain` with a `Chain` (of course with dimension corrections) and the pipeline still works.


```julia
model = Chain(Dense(3, 4), Chain((; d1=Dense(4, 4), d2=Dense(4, 4))), Dense(4, 1))

ps, st = Lux.setup(rng, model)

model(xt, ps, st)[1]
```


```
Main.ArrayAndTime{Matrix{Float32}, Float32}(Float32[-0.08124366 -1.1121564], 10.0f0)
```


# ExplicitFluxLayers

Flux by default relies on storing parameters and states in its model structs. `ExplicitFluxLayers` is an initial prototype
to make Flux explicit-parameter first.

An `AbstractExplicitLayer` is simply the minimal set of fixed attributes required to define a layer, i.e. an `AbstractExplicitLayer` is
always immutable. It doesn't contain any parameters or state variables. As an example consider `BatchNorm`

```julia
struct BatchNorm{F1,F2,F3,N} <: AbstractExplicitLayer
    λ::F1
    ϵ::N
    momentum::N
    chs::Int
    initβ::F2
    initγ::F3
    affine::Bool
    track_stats::Bool
end
```

None of these attributes of BatchNorm change over time. Next each layer needs to have the following functions defined

1. `initialparameters(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a NamedTuple containing the trainable
   parameters for the layer. For `BatchNorm`, this would contain `γ` and `β` if `affine` is set to `true` else it should
   be an empty `NamedTuple`.
2. `initialstates(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a NamedTuple containing the current
   state for the layer. For most layers this is typically empty. Layers that would potentially contain this include
   `BatchNorm`, Recurrent Neural Networks, etc. For `BatchNorm`, this would contain `μ`, `σ²`, and `training`.
3. `parameterlength(layer::CustomAbstractExplicitLayer)` & `statelength(layer::CustomAbstractExplicitLayer)` -- These can be automatically
   calculated, but it is better to define these else we construct the parameter and then count the values which is quite
   wasteful.

Additionally each AbstractExplicitLayer must return a Tuple of length 2 with the first element being the computed result and the
second element being the new state.

## Installation

```julia
] add ExplicitFluxLayers
```

## Examples
### Basic Usage

```julia
using ExplicitFluxLayers, Random, Flux, Optimisers

# Construct the layer
model = ExplicitFluxLayers.Chain(
    ExplicitFluxLayers.BatchNorm(128),
    ExplicitFluxLayers.Dense(128, 256, tanh),
    ExplicitFluxLayers.BatchNorm(256),
    ExplicitFluxLayers.Chain(
        ExplicitFluxLayers.Dense(256, 1, tanh),
        ExplicitFluxLayers.Dense(1, 10)
    )
)

# Parameter and State Variables
ps, st = ExplicitFluxLayers.setup(MersenneTwister(0), model)

# Dummy Input
x = rand(MersenneTwister(0), Float32, 128, 2);

# Run the model
y, st = ExplicitFluxLayers.apply(model, x, ps, st)

# Gradients
gs = gradient(p -> sum(ExplicitFluxLayers.apply(model, x, p, st)[1]), ps)[1]

# Optimisation
st_opt = Optimisers.setup(Optimisers.ADAM(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)
```

### Automatic Conversion of Flux Models to ExplicitFluxLayers

Call `transform` on the flux model. This will work for common cases, but all cases are not yet covered.

```julia
using ExplicitFluxLayers, Random, Metalhead

x = rand(Float32, 224, 224, 3, 1)

model = ExplicitFluxLayers.transform(ResNet18().layers[1])

ps, st = ExplicitFluxLayers.setup(MersenneTwister(0), model)

ExplicitFluxLayers.apply(model, x, ps, st)
```

### Manipulating the Parameter/State Trees

We recommend using `Functors.jl` to manipulate the data structures storing the parameter and state
variables

```julia
using Functors, ExplicitFluxLayers, Flux, Optimisers, Random

# Construct the layer
model = ExplicitFluxLayers.Chain(
    ExplicitFluxLayers.BatchNorm(128),
    ExplicitFluxLayers.Dense(128, 256, tanh),
    ExplicitFluxLayers.BatchNorm(256),
    ExplicitFluxLayers.Chain(
        ExplicitFluxLayers.Dense(256, 1, tanh),
        ExplicitFluxLayers.Dense(1, 10)
    )
)

# Parameter and State Variables
ps, st = ExplicitFluxLayers.setup(MersenneTwister(0), model)

# Let's train the Dense Layers in FP16 while keeping the BatchNorm in FP32
# The easiest way is the just pass the initW keyword but let's try the
# fancier approach
## We know that Dense Layer has (:weight, :bias) and each of them should be
## an AbstractArray
should_transform(::Any) = false
should_transform(x::NamedTuple) = all(in.((:weight, :bias), (keys(x),))) && all(Functors.isleaf, values(x))

fp16(x) = fmap(y -> Float16.(y), x)

ps = fmap(fp16, ps, exclude=should_transform)

## Let's run the model
x = rand(Float32, 128, 2);
ExplicitFluxLayers.apply(model, x, ps, st)

### NOTE: That the gradient for Dense Layers will not be in Float16
###       The updates can be made in Float32 and then converted to
###       FP16
gradient(p -> sum(ExplicitFluxLayers.apply(model, x, p, st)[1]), ps)
```

## Implemented Layers

These layers have the same API as their Flux counterparts.

* `Chain`, `Parallel`, `SkipConnection`, `BranchLayer`
* `Dense`, `Diagonal`
* `Conv`, `MaxPool`, `MeanPool`
* `BatchNorm`, `WeightNorm`, `GroupNorm`
* `ReshapeLayer`, `SelectDim`, `FlattenLayer`, `NoOpLayer`, `WrappedFunction`
* `Dropout`, `VariationalDropout`


## Cached Layers

### Some notes

1. This is currently WIP and implemented for very few layers
2. We need to define custom adjoints for cached implementations since Zygote (like most ADs) doesn't like mutations. Hence we only support inference at this point.

### Usage

```julia
using ExplicitFluxLayers, Random, Flux, Optimisers

# Construct the layer
model = ExplicitFluxLayers.Chain(
    ExplicitFluxLayers.BatchNorm(128),
    ExplicitFluxLayers.Dense(128, 256, tanh),
    ExplicitFluxLayers.BatchNorm(256),
    ExplicitFluxLayers.Chain(
        ExplicitFluxLayers.Dense(256, 1, tanh),
        ExplicitFluxLayers.Dense(1, 10)
    )
)

# Dummy Input
x = randn(Float32, 128, 10)

# Pass the input to the setup function. Now we get back the parameters, states and a cache
ps, st, cache = ExplicitFluxLayers.setup(MersenneTwister(0), model, x)

# Run the model
y, st = ExplicitFluxLayers.apply(model, x, ps, st, cache)
```

### Layers supporting caching

* `Dense` -- Only for Vectors and Matrices
* `BatchNorm` -- CUDNN version is still uncached
* `Chain`


## Benchmarks

This is mostly WIP. For some preliminary benchmarks check `benchmarks/` directory.

## TODOs

- [ ] Support Recurrent Neural Networks
- [ ] Add wider support for Flux Layers
  - [ ] Pooling -> AdaptiveMaxPool, AdaptiveMeanPool, GlobalMaxPool, GlobalMeanPool
  - [ ] Convolution --> ConvTranspose, CrossCor
  - [ ] Upsampling --> Upsample, PixelShuffle
  - [ ] General Purpose --> Maxout, Bilinear, Embedding, Dropout, AlphaDropout
  - [ ] Normalization --> LayerNorm, InstanceNorm, GroupNorm
- [ ] Port tests over from Flux

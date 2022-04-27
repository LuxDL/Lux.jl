# Lux

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![CI](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/avik-pal/Lux.jl/branch/main/graph/badge.svg?token=IMqBM1e3hz)](https://codecov.io/gh/avik-pal/Lux.jl)


The 🔥 Deep Learning Framework

## Installation

```julia
] add Lux
```

## Getting Started

```julia
using Lux, Random, Optimisers, Zygote

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

# Construct the layer
model = Chain(
    BatchNorm(128),
    Dense(128, 256, tanh),
    BatchNorm(256),
    Chain(
        Dense(256, 1, tanh),
        Dense(1, 10)
    )
)

# Parameter and State Variables
ps, st = Lux.setup(rng, model) .|> gpu

# Dummy Input
x = rand(rng, Float32, 128, 2) |> gpu

# Run the model
y, st = Lux.apply(model, x, ps, st)

# Gradients
gs = gradient(p -> sum(Lux.apply(model, x, p, st)[1]), ps)[1]

# Optimization
st_opt = Optimisers.setup(Optimisers.ADAM(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)
```

## Design Principles

* **Layers must be immutable** -- i.e. they cannot store any parameters/states but rather stores information to construct them
* **Layers return a Tuple containing the result and the updated state**
* **Layers are pure functions**
* **Given same inputs the outputs must be same** -- stochasticity is controlled by seeds passed in the state variables
* **Easily extendible for Custom Layers**: Each Custom Layer should be a subtype of either:

  a. `AbstractExplicitLayer`: Useful for Base Layers and needs to define the following functions
    1. `initialparameters(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a `ComponentArray`/`NamedTuple` containing the trainable parameters for the layer.
    2. `initialstates(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a NamedTuple containing the current state for the layer. For most layers this is typically empty. Layers that would potentially contain this include `BatchNorm`, Recurrent Neural Networks, etc.
    3. `parameterlength(layer::CustomAbstractExplicitLayer)` & `statelength(layer::CustomAbstractExplicitLayer)` -- These can be automatically calculated, but it is recommended that the user defines these.

  b. `AbstractExplicitContainerLayer`: Used when the layer is storing other `AbstractExplicitLayer`s or `AbstractExplicitContainerLayer`s. This allows good defaults of the dispatches for functions mentioned in the previous point.

## Why use Lux over Flux?

* **Large Neural Networks**
  * For small neural networks we recommend [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl).
  * For SciML Applications (Neural ODEs, Deep Equilibrium Models) solvers typically expect a monolithic parameter vector. Flux enables this via its `destructure` mechanism, however, it often leads to [weird bugs](https://github.com/FluxML/Flux.jl/issues?q=is%3Aissue+destructure). Lux forces users to make an explicit distinction between state variables and parameter variables to avoid these issues.
  * Comes battery-included for distributed training using [FluxMPI.jl](https://github.com/avik-pal/FluxMPI.jl)
* **Sensible display of Custom Layers** -- Ever wanted to see Pytorch like Network printouts or wondered how to extend the pretty printing of Flux's layers. Lux handles all of that by default.
* **Less Bug-ridden Code**
  * *No arbitrary internal mutations* -- all layers are implemented as pure functions.
  * *All layers are deterministic* given the parameter and state -- if the layer is supposed to be stochastic (say `Dropout`), the state must contain a seed which is then updated after the function call.
* **Easy Parameter Manipulation** -- Wondering why Flux doesn't have `WeightNorm`, `SpectralNorm`, etc. The implicit parameter handling makes it extremely hard to pass parameters around without mutations which AD systems don't like. With Lux implementing them is outright simple.

## Usage Examples

* Differential Equations + Deep Learning
  * [Neural ODEs for MNIST Image Classification](examples/NeuralODE/) -- Example borrowed from [DiffEqFlux.jl](https://diffeqflux.sciml.ai/dev/examples/mnist_neural_ode/)
  * [Deep Equilibrium Models](https://github.com/SciML/FastDEQ.jl)
* Image Classification
  * [ImageNet](examples/ImageNet/main.jl)
* Distributed Training using [MPI.jl](https://github.com) -- [FluxMPI](https://github.com/avik-pal/FluxMPI.jl) + [FastDEQ](https://github.com/SciML/FastDEQ.jl/examples)

## Recommended Libraries for Various ML Tasks

Lux is exclusively focused on designing Neural Network Architectures. All other parts of the DL training/evaluation pipeline should be offloaded to the following frameworks:

* Data Manipulation/Loading -- [Augmentor.jl](https://evizero.github.io/Augmentor.jl/stable/), [DataLoaders.jl](https://lorenzoh.github.io/DataLoaders.jl/docs/dev/), [Images.jl](https://juliaimages.org/stable/)
* Optimisation -- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl), [ParameterSchedulers.jl](https://darsnack.github.io/ParameterSchedulers.jl/dev/README.html)
* Automatic Differentiation -- [Zygote.jl](https://github.com/FluxML/Zygote.jl)
* Parameter Manipulation -- [Functors.jl](https://fluxml.ai/Functors.jl/stable/)
* Model Checkpointing -- [Serialization.jl](https://docs.julialang.org/en/v1/stdlib/Serialization/)
* Activation Functions / Common Neural Network Primitives -- [NNlib.jl](https://fluxml.ai/Flux.jl/stable/models/nnlib/)
* Distributed Training -- [FluxMPI.jl](https://github.com/avik-pal/FluxMPI.jl)
* Training Visualization -- [Wandb.jl](https://github.com/avik-pal/Wandb.jl)

If you found any other packages useful, please open a PR and add them to this list.

## Implemented Layers

We don't have a Documentation Page as of now. But all these functions have docs which can be access in the REPL help mode.

* `Chain`, `Parallel`, `SkipConnection`, `BranchLayer`, `PairwiseFusion`
* `Dense`, `Diagonal`
* `Conv`, `MaxPool`, `MeanPool`, `GlobalMaxPool`, `GlobalMeanPool`, `Upsample`, `AdaptiveMaxPool`, `AdaptiveMeanPool`
* `BatchNorm`, `WeightNorm`, `GroupNorm`
* `ReshapeLayer`, `SelectDim`, `FlattenLayer`, `NoOpLayer`, `WrappedFunction`
* `Dropout`, `VariationalHiddenDropout`


## TODOs

- [ ] Support Recurrent Neural Networks
- [ ] Add wider support for Flux Layers
  - [ ] Convolution --> ConvTranspose, CrossCor
  - [ ] Upsampling --> PixelShuffle
  - [ ] General Purpose --> Maxout, Bilinear, Embedding, AlphaDropout
  - [ ] Normalization --> LayerNorm, InstanceNorm
- [ ] Port tests over from Flux

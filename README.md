# ExplicitFluxLayers

## Design Principles

Flux by default relies on storing parameters and states in its model structs. `ExplicitFluxLayers` is makes Flux explicit-parameter first. 

An `AbstractExplicitLayer` is the minimal set of fixed attributes required to define a layer. It must always be immutable. It doesn't contain any parameters or state variables, but defines how to construct them. As an example consider `BatchNorm`

```julia
# Ignore the syntax it is just for demonstration
struct BatchNorm{affine,track_stats,F1,F2,F3,N} <: AbstractNormalizationLayer{affine,track_stats} <: AbstractExplicitLayer
    λ::F1
    ϵ::N
    momentum::N
    chs::Int
    initβ::F2
    initγ::F3
end
```

None of these attributes of BatchNorm can change over time.

Next each layer needs to have the following functions defined

1. `initialparameters(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a `ComponentArray` containing the trainable parameters for the layer. For `BatchNorm`, this would contain `γ` and `β` if `affine` is set to `true` else it should be a `ComponentArray` with fields `γ` and `β` set to `nothing`.
2. `initialstates(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a NamedTuple containing the current state for the layer. For most layers this is typically empty. Layers that would potentially contain this include `BatchNorm`, Recurrent Neural Networks, etc. For `BatchNorm`, this would contain `μ`, `σ²`, and `training`.
3. `parameterlength(layer::CustomAbstractExplicitLayer)` & `statelength(layer::CustomAbstractExplicitLayer)` -- These can be automatically calculated, but it is better to define these else we construct the parameter and then count the values which is quite wasteful.

Additionally, each AbstractExplicitLayer must return a Tuple of length 2 with the first element being the computed result and the second element being the new state. Finally, we recommend users to write to not mutate `ps` and `st` when the layers are called. Instead update the state by returning a new NamedTuple.

## Installation

```julia
] add ExplicitFluxLayers
```

## Why use ExplicitFluxLayers over Flux?

* When using Large Neural Networks (for small networks see [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl)) for SciML Applications (Neural ODEs, Deep Equilibrium Models) -- Solvers typically expect a monolithic parameter vector. Flux enables this via its `destructure` mechanism, however, it often leads to [weird bugs](https://github.com/FluxML/Flux.jl/issues?q=is%3Aissue+destructure). EFL forces users to make an explicit distinction between state variables and parameter variables to avoid these issues.
* Layers must be properly subtyped which allows easy initialization (like `Flux.@functor`) but additionally allows proper dispatch for pretty printing arbitrary user models.
* No arbitrary internal mutations -- all layers are implemented as pure functions (which means there can be a minor overhead compared to Flux).
* All layers are deterministic given the parameter and state -- if the layer is supposed to be stochastic (say `Dropout`), the state must contain a seed which is then updated after the function call.
* Returning `state` by each layer makes it easy to RNNs (I promise to implement them soonish).
* Need to manipulate parameters before passing them to other layers -- `WeightNorm`, `SpectralNorm`

## Why not use ExplicitFluxLayers?

* Both EFL and Flux use the same backend so performance should not be the motivating reason to shift (it should also not deter the intention to migrate).
* EFL is verbose and it is meant to be that way. Explicitly passing PRNGKeys and explicitly handing parameter and state variables might not be very user friendly and most people might never need it. However, this design aspect makes code less bug ridden and easy to debug/reproduce.
* Flux is maintained by a larger community while EFL is currently being maintained only by me.
* Higher Order derivatives are completely broken -- for Flux certain things work but for EFL I can assure you nothing works (mostly Zygote and ComponentArrays/NamedTuple compatibility issues).

## Getting Started
### Basic Usage

```julia
using ExplicitFluxLayers, Random, Flux, Optimisers

# Construct the layer
model = EFL.Chain(
    EFL.BatchNorm(128),
    EFL.Dense(128, 256, tanh),
    EFL.BatchNorm(256),
    EFL.Chain(
        EFL.Dense(256, 1, tanh),
        EFL.Dense(1, 10)
    )
)

# Parameter and State Variables
ps, st = EFL.setup(MersenneTwister(0), model)

# Dummy Input
x = rand(MersenneTwister(0), Float32, 128, 2);

# Run the model
y, st = EFL.apply(model, x, ps, st)

# Gradients
gs = gradient(p -> sum(EFL.apply(model, x, p, st)[1]), ps)[1]

# Optimisation
st_opt = Optimisers.setup(Optimisers.ADAM(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)
```

### Automatic Conversion of Flux Models to ExplicitFluxLayers

Call `transform` on the flux model. This will work for common cases, but all cases are not yet covered.

```julia
using ExplicitFluxLayers, Random, Metalhead

x = rand(Float32, 224, 224, 3, 1)

model = EFL.transform(ResNet18().layers[1])

ps, st = EFL.setup(MersenneTwister(0), model)

EFL.apply(model, x, ps, st)
```

## Usage Examples

* [Neural ODEs for MNIST Image Classification](examples/NeuralODE/neural_ode.jl) -- Example borrowed from [DiffEqFlux.jl](https://diffeqflux.sciml.ai/dev/examples/mnist_neural_ode/)
* [Deep Equilibrium Models](https://github.com/SciML/FastDEQ.jl)

## Implemented Layers

These layers have the same API as their Flux counterparts.

* `Chain`, `Parallel`, `SkipConnection`, `BranchLayer`, `PairwiseFusion`
* `Dense`, `Diagonal`
* `Conv`, `MaxPool`, `MeanPool`, `GlobalMaxPool`, `GlobalMeanPool`, `Upsample`
* `BatchNorm`, `WeightNorm`, `GroupNorm`
* `ReshapeLayer`, `SelectDim`, `FlattenLayer`, `NoOpLayer`, `WrappedFunction`
* `Dropout`, `VariationalDropout`


## Benchmarks

This is mostly WIP. For some preliminary benchmarks check `benchmarks/` directory.

## TODOs

- [ ] Support Recurrent Neural Networks
- [ ] Add wider support for Flux Layers
  - [ ] Pooling -> AdaptiveMaxPool, AdaptiveMeanPool
  - [ ] Convolution --> ConvTranspose, CrossCor
  - [ ] Upsampling --> PixelShuffle
  - [ ] General Purpose --> Maxout, Bilinear, Embedding, AlphaDropout
  - [ ] Normalization --> LayerNorm, InstanceNorm
- [ ] Port tests over from Flux

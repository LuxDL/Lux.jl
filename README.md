# ExplicitFluxLayers

Flux by default relies on storing parameters and states in its model structs. `ExplicitLayers` is an initial prototype
to make Flux explicit-parameter first.

An `ExplicitLayer` is simply the minimal set of fixed attributes required to define a layer, i.e. an `ExplicitLayer` is
always immutable. It doesn't contain any parameters or state variables. As an example consider `BatchNorm`

```julia
struct BatchNorm{F1,F2,F3,N} <: ExplicitLayer
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

1. `initialparameters(rng::AbstractRNG, layer::CustomExplicitLayer)` -- This returns a NamedTuple containing the trainable
   parameters for the layer. For `BatchNorm`, this would contain `γ` and `β` if `affine` is set to `true` else it should
   be an empty `NamedTuple`.
2. `initialstates(rng::AbstractRNG, layer::CustomExplicitLayer)` -- This returns a NamedTuple containing the current
   state for the layer. For most layers this is typically empty. Layers that would potentially contain this include
   `BatchNorm`, Recurrent Neural Networks, etc. For `BatchNorm`, this would contain `μ`, `σ²`, and `training`.
3. `parameterlength(layer::CustomExplicitLayer)` & `statelength(layer::CustomExplicitLayer)` -- These can be automatically
   calculated, but it is better to define these else we construct the parameter and then count the values which is quite
   wasteful.

Additionally each ExplicitLayer must return a Tuple of length 2 with the first element being the computed result and the
second element being the new state.

## Usage

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

## Currently Implemented Explicit Layers (none of these are exported)

These layers have the same API as their Flux counterparts.

* `Chain`
* `Dense`
* `Conv`
* `BatchNorm`
* `WeightNorm`
* `Parallel`
* `SkipConnection`

# Why we wrote Lux?

Julia already has quite a few well established Neural Network Frameworks --
[Flux](https://fluxml.ai/) & [KNet](https://denizyuret.github.io/Knet.jl/latest/). However,
certain design elements -- **Coupled Model and Parameters** & **Internal Mutations** --
associated with these frameworks make them less compiler and user friendly. Making changes
to address these problems in the respective frameworks would be too disruptive for users.
Here comes in `Lux`: a neural network framework built completely using pure functions to make
it both compiler and autodiff friendly.

## Design Principles

* **Layers must be immutable** -- cannot store any parameter/state but rather store the
  information to construct them
* **Layers are pure functions**
* **Layers return a Tuple containing the result and the updated state**
* **Given same inputs the outputs must be same** -- yes this must hold true even for
  stochastic functions. Randomness must be controlled using `rng`s passed in the state.
* **Easily extensible**
* **Extensive Testing** -- All layers and features are tested across all supported AD
  backends across all supported hardware backends.

## Why use Lux over Flux?

* **Neural Networks for SciML**: For SciML Applications (Neural ODEs, Deep Equilibrium
  Models) solvers typically expect a monolithic parameter vector. Flux enables this via its
  `destructure` mechanism, but `destructure` comes with various
  [edge cases and limitations](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.destructure). Lux
  forces users to make an explicit distinction between state variables and parameter
  variables to avoid these issues. Also, it comes battery-included for distributed training.

* **Sensible display of Custom Layers** -- Ever wanted to see Pytorch like Network printouts
  or wondered how to extend the pretty printing of Flux's layers? Lux handles all of that
  by default.

* **Truly immutable models** - No *unexpected internal mutations* since all layers are
  implemented as pure functions. All layers are also *deterministic* given the parameters
  and state: if a layer is supposed to be stochastic (say [`Dropout`](@ref)), the state
  must contain a seed which is then updated after the function call.

* **Easy Parameter Manipulation** -- By separating parameter data and layer structures,
  Lux makes implementing [`WeightNorm`](@ref), `SpectralNorm`, etc. downright trivial.
  Without this separation, it is much harder to pass such parameters around without
  mutations which AD systems don't like.

* **Wider AD Support** -- Lux has extensive support for most
  [AD systems in julia](@ref autodiff-lux), while Flux is mostly tied to Zygote (with some
  initial support for Enzyme).

* **Small Neural Networks on CPU** -- Lux is developed for training large neural networks.
  For smaller architectures, we recommend using
  [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) or even better use it in
  conjunction with Lux via [`ToSimpleChainsAdaptor`](@ref).

* **Reliability** -- We have learned from the mistakes of the past with Flux and everything
  in our core framework is extensively tested, along with downstream CI to ensure that
  everything works as expected.

!!! tip "Revising Previous Recommendation about Large Models"

    Previously we recommended not using Lux for very large models. But we have been making
    a lot of head-way with [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and it
    would be worthwhile to test larger models with Lux. See [compiling Lux
    models](@ref reactant-compilation) for more information.

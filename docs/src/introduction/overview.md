# Why we wrote Lux?

Julia already has quite a few well established Neural Network Frameworks --
[Flux](https://fluxml.ai/) & [KNet](https://denizyuret.github.io/Knet.jl/latest/). However,
certain design elements -- **Coupled Model and Parameters** & **Internal Mutations** --
associated with these frameworks make them less compiler and user friendly. Making changes
to address these problems in the respective frameworks would be too disruptive for users.
Here comes in `Lux`: a neural network framework built completely using pure functions to make
it both compiler and autodiff friendly.

## Performance and Deployment with Reactant

Lux.jl takes a **Reactant-first approach** to deliver exceptional performance and seamless deployment capabilities:

- **XLA Compilation** -- Lux models compile to highly optimized XLA code via [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), delivering significant speedups on CPU, GPU, and TPU.

- **Cross-Platform Performance** -- Run the same Lux model with optimal performance across different hardware backends (CPU, NVIDIA GPUs, AMD GPUs, TPUs) without code changes, simply by switching the Reactant backend.

- **Production Deployment** -- Compiled models can be exported and deployed to production servers and edge devices by leveraging the rich TensorFlow ecosystem, making Lux suitable for real-world applications.

- **Large Model Support** -- With Reactant compilation, Lux now excels at training very large models that were previously challenging, making it competitive with other frameworks for large-scale deep learning.

## Design Principles

- **Layers must be immutable** -- cannot store any parameter/state but rather store the
  information to construct them
- **Layers are pure functions**
- **Layers return a Tuple containing the result and the updated state**
- **Given same inputs the outputs must be same** -- yes this must hold true even for
  stochastic functions. Randomness must be controlled using `rng`s passed in the state.
- **Easily extensible**
- **Extensive Testing** -- All layers and features are tested across all supported AD
  backends across all supported hardware backends.

## Why use Lux over Flux?

- **High-Performance XLA Compilation** -- Lux's Reactant-first approach enables XLA compilation for dramatic performance improvements across CPU, GPU, and TPU. Models compile to highly optimized code that eliminates Julia overhead and leverages hardware-specific optimizations.

- **Production-Ready Deployment** -- Deploy Lux models to production environments using the mature TensorFlow ecosystem. Compiled models can be exported and run on servers, edge devices, and mobile platforms.

- **Neural Networks for SciML**: For SciML Applications (Neural ODEs, Deep Equilibrium
  Models) solvers typically expect a monolithic parameter vector. Flux enables this via its
  `destructure` mechanism, but `destructure` comes with various
  [edge cases and limitations](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.destructure). Lux
  forces users to make an explicit distinction between state variables and parameter
  variables to avoid these issues. Also, it comes battery-included for distributed training.

- **Sensible display of Custom Layers** -- Ever wanted to see Pytorch like Network printouts
  or wondered how to extend the pretty printing of Flux's layers? Lux handles all of that
  by default.

- **Truly immutable models** - No _unexpected internal mutations_ since all layers are
  implemented as pure functions. All layers are also _deterministic_ given the parameters
  and state: if a layer is supposed to be stochastic (say [`Lux.Dropout`](@ref)), the state
  must contain a seed which is then updated after the function call.

- **Easy Parameter Manipulation** -- By separating parameter data and layer structures,
  Lux makes implementing [`WeightNorm`](@ref), `SpectralNorm`, etc. downright trivial.
  Without this separation, it is much harder to pass such parameters around without
  mutations which AD systems don't like.

- **Wider AD Support** -- Lux has extensive support for most
  [AD systems in julia](@ref autodiff-lux), while Flux is mostly tied to Zygote (with some
  initial support for Enzyme).

- **Optimized for All Model Sizes** -- Whether you're working with small prototypes or large production models, Lux delivers optimal performance. For the smallest networks where minimal overhead is critical, you can use [`ToSimpleChainsAdaptor`](@ref) to leverage SimpleChains.jl's specialized CPU optimizations.

- **Reliability** -- We have learned from the mistakes of the past with Flux and everything
  in our core framework is extensively tested, along with downstream CI to ensure that
  everything works as expected.

# Why we wrote Lux?

Julia already has quite a few well established Neural Network Frameworks --
[Flux](https://fluxml.ai/) & [KNet](https://denizyuret.github.io/Knet.jl/latest/). However,
certain design elements -- **Coupled Model and Parameters** & **Internal Mutations** --
associated with these frameworks make them less compiler and user friendly. Making changes
to address these problems in the respective frameworks would be too disruptive for users.
Here comes in `Lux` a neural network framework built completely using pure functions to make
it both compiler and autodiff friendly.

# Design Principles

* **Layers must be immutable** -- cannot store any parameter/state but rather store the
  information to construct them
* **Layers are pure functions**
* **Layers return a Tuple containing the result and the updated state**
* **Given same inputs the outputs must be same** -- yes this must hold true even for
  stochastic functions. Randomness must be controlled using `rng`s passed in the state.
* **Easily extendible**

# Why use Lux over Flux?

* **Neural Networks for SciML**: For SciML Applications (Neural ODEs, Deep Equilibrium
  Models) solvers typically expect a monolithic parameter vector. Flux enables this via its
  `destructure` mechanism, however, it often leads to
  [weird bugs](https://github.com/FluxML/Flux.jl/issues?q=is%3Aissue+destructure). Lux
  forces users to make an explicit distinction between state variables and parameter
  variables to avoid these issues. Also, it comes battery-included for distributed training
  using [FluxMPI.jl](https://github.com/avik-pal/FluxMPI.jl) *(I know :P the naming)*
  
* **Sensible display of Custom Layers** -- Ever wanted to see Pytorch like Network printouts
  or wondered how to extend the pretty printing of Flux's layers. Lux handles all of that
  by default.
  
* **Less Bug-ridden Code** - *No arbitrary internal mutations* since all layers are
  implemented as pure functions. *All layers are deterministic* given the parameter and
  state (if the layer is supposed to be stochastic (say `Dropout`), the state must contain a
  seed which is then updated after the function call).

* **Easy Parameter Manipulation** -- Wondering why Flux doesn't have `WeightNorm`,
  `SpectralNorm`, etc. The implicit parameter handling makes it extremely hard to pass
  parameters around without mutations which AD systems don't like. With Lux implementing
  them is outright simple.


# Why not use Lux?

* **Small Neural Networks on CPU** -- Lux is developed for training large neural networks.
  For smaller architectures, we recommend using
  [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl).

* **Lux won't magically speed up your code (yet)** -- Lux shares the same backend with Flux
  and so if your primary desire to shift is driven by performance, you will be disappointed.

* **Special Architecture Support** -- Unfortunately, we currently don't support Cloud TPUs
  and even AMD GPUs are not well tested. (We do plan to support these in the nearish
  future)

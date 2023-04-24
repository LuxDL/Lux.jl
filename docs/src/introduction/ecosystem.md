# Ecosystem

## Frameworks extending Lux

- [Boltz.jl](https://github.com/avik-pal/Lux.jl/tree/main/lib/Boltz) -- Prebuilt deep learning models for image classification tasks
- [DeepEquilibriumNetworks.jl](https://github.com/SciML/DeepEquilibriumNetworks.jl) -- Continuous and Discrete Deep Equilibrium Networks
- [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) -- Neural Differential Equations, Continuous Normalizing Flows, etc.

## Extended Julia Ecosystem

As you might have noticed we don't do much apart from Neural Networks. All other parts of the DL training/evaluation pipeline should be offloaded to:

### Automatic Differentiation

- [Zygote.jl](https://github.com/FluxML/Zygote.jl) -- Currently the default and recommended AD library
- [Tracker.jl](https://github.com/FluxML/Tracker.jl) -- Well tested and robust AD library (might fail on edge cases)
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) -- (Very) Experimental Support
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) -- For forward mode AD support
- [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) -- Tape based reverse mode AD (might fail on edge cases and doesn't work on GPU)

### Data Manipulation and Loading

- [Augmentor.jl](https://evizero.github.io/Augmentor.jl/stable/)
- [DataLoaders.jl](https://lorenzoh.github.io/DataLoaders.jl/docs/dev/)
- [Images.jl](https://juliaimages.org/stable/)
- [DataAugmentation.jl](https://lorenzoh.github.io/DataAugmentation.jl/dev/README.md.html)

### Distributed DataParallel Training

- [FluxMPI.jl](https://github.com/avik-pal/FluxMPI.jl)

### Neural Network Primitives

- [NNlib.jl](https://fluxml.ai/Flux.jl/stable/models/nnlib/)
- [LuxLib.jl](https://luxdl.github.io/LuxLib.jl/dev/)

### Optimization 

- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl)
- [ParameterSchedulers.jl](https://darsnack.github.io/ParameterSchedulers.jl/dev/README.html)
- [Optimization.jl](http://optimization.sciml.ai/stable/)

### Parameter Manipulation

- [Functors.jl](https://fluxml.ai/Functors.jl/stable/)


### Serialization

- [Serialization.jl](https://docs.julialang.org/en/v1/stdlib/Serialization/)
- [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)

### Testing Utilities

- [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl) -- Finite Differencing. Useful for testing gradient correctness
- [JET.jl](https://aviatesk.github.io/JET.jl/dev/)
- [LuxTestUtils.jl](https://github.com/LuxDL/LuxTestUtils.jl)

### Training Visualization & Logging

- [Wandb.jl](https://github.com/avik-pal/Wandb.jl)
- [TensorBoardLogger.jl](https://github.com/PhilipVinc/TensorBoardLogger.jl)
